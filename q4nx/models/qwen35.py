from pprint import pp

from ..model_converter import __Q4NX_Converter
from ..constants import ModelArch
from gguf import GGUFReader, dequantize, quantize, GGMLQuantizationType
from safetensors.torch import save_file
from einops import rearrange, repeat
import torch
import json

class Qwen35(__Q4NX_Converter, model_arch=ModelArch.QWEN35_4B):
    def __init__(self, source, config_json_path=None):
        print("[INFO] Using Qwen35_4B converter")
        self.gguf_reader = None
        self.gguf_tensors = []
        self.hf_source = None
        self.hf_dir = None
        self.weight_map = {}
        self.hf_shards = {}
        if isinstance(source, GGUFReader):
            self.gguf_reader = source
            self.gguf_tensors = {t.name: t for t in source.tensors}
            self.initialize()
        else:
            self.hf_source = source
            self.hf_dir = self._resolve_source(source)
            self.initialize(config_json_path=config_json_path)

    def initialize(self, config_json_path=None):
        super().initialize()

    def convert(self, q4nx_path: str, weights_type: str = 'language'):
        self.q4nx_tensors = {}
        if self.gguf_reader is not None:
            self._convert_gguf(q4nx_path, weights_type)
        else:
            self._convert_hf(q4nx_path, weights_type)

    def _convert_gguf(self, q4nx_path: str, weights_type: str):
        if weights_type == "language":
            reorder_linear_required = True
            if self.gguf_reader.fields["qwen35.feed_forward_length"].contents() <= 6144:
                reorder_linear_required = False
            if reorder_linear_required:
                print("[INFO] Reorder linear required!")

            full_attntion_interval = self.gguf_reader.fields["qwen35.full_attention_interval"].contents()            
            if not self._has_lm_head():
                print("[INFO] Model does not have a lm_head, use embedding weights as lm_head")
                unpacked = self.gguf_tensors["token_embd.weight"].unpack(GGMLQuantizationType.Q8_0)
                target_dtype = self.gguf_tensors["token_embd.weight"].get_used_quantization_type(GGMLQuantizationType.Q8_0)
                self.q4nx_tensors["lm_head.weight"] = self._pack(*unpacked, tensor_type=target_dtype)

            for key, gguf_tensor in self.gguf_tensors.items():
                if ".nextn." in gguf_tensor.name:
                    print(f"[SKIP] {gguf_tensor.name} (MTP next-token prediction weights, absent from official Q4NX)")
                    continue
                target_dtype = gguf_tensor.get_used_quantization_type(self.tensor_q4nx_type_map[gguf_tensor.name])
                print(f"Processing tensor: {gguf_tensor.name} with type {gguf_tensor.tensor_type.name} -> {self.forward_name_map[gguf_tensor.name]} with dtype {target_dtype.name}")
                if "token_embd.weight" in gguf_tensor.name:
                    w = dequantize(gguf_tensor.data, gguf_tensor.tensor_type)
                    w = torch.from_numpy(w).contiguous().to(torch.bfloat16)
                    self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = w
                    continue
                
                new_name = self.forward_name_map[gguf_tensor.name]
                layer_id = 0
                if "layers." in new_name:
                    layer_id = int(new_name.split("layers.")[1].split(".")[0])

                unpacked = gguf_tensor.unpack(target_dtype)

                if layer_id % full_attntion_interval == (full_attntion_interval - 1):    
                    if "q_proj" in self.forward_name_map[gguf_tensor.name]:
                        print("[INFO] Seperate q, gate for q_proj")
                        DH = self.gguf_reader.fields["qwen35.attention.value_length"].contents()
                        d, m, qw = unpacked
                        d = rearrange(d, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        m = rearrange(m, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        qw = rearrange(qw, '(g p h) c -> (p g h) c', p = 2, h = DH).contiguous()
                        unpacked = (d, m, qw)

                else:
                    if "self_attn.gate_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print("[INFO] Reorder Gate")
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            d, m, qw = unpacked
                            d = rearrange(d, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            m = rearrange(m, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            qw = rearrange(qw, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                            unpacked = (d, m, qw)

                    if "qkv_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print("[INFO] Seperate q, gate for q_proj")
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            d, m, qw = unpacked
                            d0, d1 = d.chunk(2, dim = 0)
                            m0, m1 = m.chunk(2, dim = 0)
                            qw0, qw1 = qw.chunk(2, dim = 0)
                            print(d0.shape, d1.shape, m0.shape, m1.shape, qw0.shape, qw1.shape)
                            pp = DH

                            d1 = rearrange(d1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()
                            m1 = rearrange(m1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()
                            qw1 = rearrange(qw1, '(q g p) c -> (g q p) c', p = pp, q = 2).contiguous()

                            d = torch.cat([d0, d1], dim = 0).contiguous()
                            m = torch.cat([m0, m1], dim = 0).contiguous()
                            qw = torch.cat([qw0, qw1], dim = 0).contiguous()
                            unpacked = (d, m, qw)

                    if "ssm_out_proj" in self.forward_name_map[gguf_tensor.name]:
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d, m, qw = unpacked
                            DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                            DH = DH // 32
                            BLOCK_SIZE = 32
                            d = rearrange(d, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            m = rearrange(m, 'r (q g p) -> r (g q p)', p = DH, q = 2).contiguous()
                            qw = rearrange(qw, 'r (q g p) -> r (g q p)', p = DH * BLOCK_SIZE, q = 2).contiguous()

                            unpacked = (d, m, qw)

                    if "ssm_alpha_proj" in self.forward_name_map[gguf_tensor.name] or "ssm_beta_proj" in self.forward_name_map[gguf_tensor.name]:
                        d, m, qw = unpacked
                        w = gguf_tensor.dequantize()
                        if reorder_linear_required:
                            w = rearrange(w, '(q g) c -> (g q) c', q = 2).contiguous()

                        new_name = self.forward_name_map[gguf_tensor.name]
                        new_name = new_name.replace("alpha_proj", "alpha_proj.bf16").replace("beta_proj", "beta_proj.bf16")
                        self.q4nx_tensors[new_name] = w
                        if reorder_linear_required:
                            print(f"[INFO] Reorder for {self.forward_name_map[gguf_tensor.name]}")
                            d = rearrange(d, '(q g) c -> (g q) c', q = 2).contiguous()
                            m = rearrange(m, '(q g) c -> (g q) c', q = 2).contiguous()
                            qw = rearrange(qw, '(q g) c -> (g q) c', q = 2).contiguous()

                        if (d.shape[0] < 32):
                            d = repeat(d, 'd c -> (r d) c', r = 2).contiguous()
                            m = repeat(m, 'd c -> (r d) c', r = 2).contiguous()
                            qw = repeat(qw, 'd c -> (r d) c', r = 2).contiguous()

                        unpacked = (d, m, qw)


                    if "ssm_conv1d" in self.forward_name_map[gguf_tensor.name]:
                        print("[INFO] transpose conv1d")

                        DH = self.gguf_reader.fields["qwen35.ssm.state_size"].contents()
                        d = unpacked[0]
                        
                        if reorder_linear_required:
                            d0, d1 = d.chunk(2, dim = 0)

                            d1 = rearrange(d1, '(q g p) c -> (g q p) c', p = DH, q = 2).contiguous()
                        
                            d = torch.cat([d0, d1], dim = 0).contiguous()
                        d = d.T.contiguous()
                        unpacked = [d]
                
                    if "ssm_a" in gguf_tensor.name[-5:]:
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = rearrange(val, '(q g) -> (g q)', q = 2).contiguous()
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

                    if "ssm_dt" in gguf_tensor.name:
                        val = unpacked[0].to(torch.float32).contiguous()
                        if reorder_linear_required:
                            val = rearrange(val, '(q g) -> (g q)', q = 2).contiguous()
                        self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = val
                        continue

                self.q4nx_tensors[self.forward_name_map[gguf_tensor.name]] = self._pack(*unpacked, tensor_type=target_dtype)
            self._extract_tokenizer_json(q4nx_path)                
        elif weights_type == "vision":
            for key, gguf_tensor in self.gguf_tensors.items():
                unpacked = gguf_tensor.unpack(GGMLQuantizationType.BF16)
                assert len(unpacked) == 1
                assert type(unpacked[0]) == torch.Tensor, "Vision model tensors"
                weights = unpacked[0]
                if weights.dtype != torch.bfloat16:
                    weights = weights.to(torch.bfloat16)
                    
                new_name = self.forward_name_map[gguf_tensor.name]                        
                
                if new_name.endswith("fc2.weight") or new_name.endswith("fc1.weight")\
                    or new_name.endswith("attn.proj.weight") or new_name.endswith("attn.qkv.weight"):
                    weights = self.vision_mm_weight_rearrange(weights)
                
                self.q4nx_tensors[new_name] = weights
                
            combined_patched_embeding= torch.stack(
                [self.q4nx_tensors["model.visual.patch_embed.proj.weight"],
                         self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
                 ], dim=2
            )
            del self.q4nx_tensors["model.visual.patch_embed.proj.weight"]
            del self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
            self.q4nx_tensors["model.visual.patch_embed.proj.weight"] = combined_patched_embeding
    
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35 model")

        self._export_weights(q4nx_path, weights_type)

    def _convert_hf(self, q4nx_path: str, weights_type: str):
        if weights_type == "language":
            self._convert_hf_language(q4nx_path)
        elif weights_type == "vision":
            self._convert_hf_vision(q4nx_path)
        else:
            raise ValueError(f"Unsupported weights_type: {weights_type} for Qwen35 HF conversion")

    def _convert_hf_language(self, q4nx_path: str):
        import re
        self.q4nx_tensors = {}
        # Build HF name map with {bid} expanded to actual layer numbers.
        # HF names have prefix model.language_model.layers.{bid}.X
        # Q4NX names have prefix model.layers.{bid}.X
        # After stripping model.language_model. from HF, key is layers.{bid}.X
        # So we need Q4NX names with model. prefix stripped for {bid} entries.
        hf_name_map = {}
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name:
                continue
            if "{bid}" in q4nx_name:
                # Strip 'model.' prefix so it matches stripped HF key (layers.{bid}.X)
                stripped = q4nx_name.replace("model.", "", 1)
                pattern = re.escape(stripped).replace(r"\{bid\}", r"(\d+)")
                found = sorted(set(
                    int(m.group(1))
                    for n in self.weight_map
                    if (m := re.match("^" + pattern + "$", n.replace("model.language_model.", "")))
                ))
                for bid in found:
                    hf_key = stripped.format(bid=bid)
                    q4nx_key = q4nx_name.format(bid=bid)
                    hf_name_map[hf_key] = q4nx_key
            else:
                hf_name_map[q4nx_name] = q4nx_name

        config_path = self.hf_dir / "config.json"
        head_dim = None
        ssm_state_size = None
        if config_path.is_file():
            with open(config_path) as f:
                cfg = json.load(f)
            head_dim = cfg.get("head_dim") or cfg.get("attention_value_length")
            ssm_state_size = cfg.get("ssm_state_size") or cfg.get("conv_kernel_size")

        reorder_linear = ssm_state_size is not None and ssm_state_size > 6144

        for name in sorted(self.weight_map):
            key = name.replace("model.language_model.", "")
            if ".nextn." in key:
                continue
            if key not in hf_name_map:
                print(f"[WARN] Unmapped HF tensor: {name}")
                continue
            w = self._load_tensor(name)
            q4nx_name = hf_name_map[key]

            if key == "model.embed_tokens.weight":
                self.q4nx_tensors[q4nx_name] = w.to(torch.bfloat16)
                continue
            if key == "lm_head.weight":
                self.q4nx_tensors[q4nx_name] = w
                continue

            if "self_attn.q_proj.weight" in key and head_dim is not None:
                w = rearrange(w, '(g p h) c -> (p g h) c', p=2, h=head_dim).contiguous()

            if reorder_linear:
                if "linear_attn.in_proj_qkv.weight" in key and ssm_state_size is not None:
                    d_half = w.shape[0] // 2
                    w0 = w[:d_half]
                    w1 = w[d_half:]
                    w1 = rearrange(w1, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()
                    w = torch.cat([w0, w1], dim=0).contiguous()

                if "self_attn.gate_proj.weight" in key and ssm_state_size is not None:
                    w = rearrange(w, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()

                if "linear_attn.ssm_out_proj.weight" in key and ssm_state_size is not None:
                    DH = ssm_state_size // 32
                    BLOCK_SIZE = 32
                    w = rearrange(w, 'r (q g p) -> r (g q p)', p=DH, q=2).contiguous()

                if "linear_attn.ssm_alpha_proj.weight" in key or "linear_attn.ssm_beta_proj.weight" in key:
                    w = rearrange(w, '(q g) c -> (g q) c', q=2).contiguous()
                    bf16_name = q4nx_name.replace("alpha_proj", "alpha_proj.bf16").replace("beta_proj", "beta_proj.bf16")
                    self.q4nx_tensors[bf16_name] = w
                    if w.shape[0] < 32:
                        w = repeat(w, 'd c -> (r d) c', r=2).contiguous()

                if "linear_attn.ssm_conv1d.weight" in key and ssm_state_size is not None:
                    d_half = w.shape[0] // 2
                    w0 = w[:d_half]
                    w1 = w[d_half:]
                    w1 = rearrange(w1, '(q g p) c -> (g q p) c', p=ssm_state_size, q=2).contiguous()
                    w = torch.cat([w0, w1], dim=0).contiguous()
                    w = w.T.contiguous()

                if "linear_attn.ssm_a" in key and key.endswith("ssm_a"):
                    w = w.float()
                    w = rearrange(w, '(q g) -> (g q)', q=2).contiguous()

                if "linear_attn.ssm_dt.bias" in key:
                    w = w.float()
                    w = rearrange(w, '(q g) -> (g q)', q=2).contiguous()

            self.q4nx_tensors[q4nx_name] = w

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX tensors")
        self._export_weights(q4nx_path, "language")

    def _convert_hf_vision(self, q4nx_path: str):
        import re
        self.q4nx_tensors = {}
        # Build a proper HF name map with {bid} expanded to actual layer numbers.
        hf_name_map = {}
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name or "{bid}" not in q4nx_name:
                continue
            # Detect how many layers exist for this pattern from weight_map
            pattern = re.escape(q4nx_name).replace(r"\{bid\}", r"(\d+)")
            found = sorted(set(
                int(m.group(1))
                for n in self.weight_map
                if (m := re.match("^" + pattern + "$", n))
            ))
            for bid in found:
                concrete = q4nx_name.format(bid=bid)
                hf_name_map[concrete] = concrete
        # Add non-bid entries (patch_embed, merger, etc.)
        for param_info in self.q4nx_config["name_map"].values():
            q4nx_name = param_info["q4nx_name"]
            if "rope_freqs" in q4nx_name or "{bid}" in q4nx_name:
                continue
            hf_name_map[q4nx_name] = q4nx_name

        for name in sorted(self.weight_map):
            if name not in hf_name_map:
                continue
            w = self._load_tensor(name)
            if w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            q4nx_name = hf_name_map[name]

            if q4nx_name.endswith("linear_fc2.weight") or q4nx_name.endswith("linear_fc1.weight") \
                or q4nx_name.endswith("attn.proj.weight") or q4nx_name.endswith("attn.qkv.weight"):
                w = self.vision_mm_weight_rearrange(w)

            self.q4nx_tensors[q4nx_name] = w

        combined_patched_embeding = torch.stack(
            [self.q4nx_tensors["model.visual.patch_embed.proj.weight"],
             self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
            ], dim=2
        )
        del self.q4nx_tensors["model.visual.patch_embed.proj.weight"]
        del self.q4nx_tensors["model.visual.patch_embed.proj.weight.1"]
        self.q4nx_tensors["model.visual.patch_embed.proj.weight"] = combined_patched_embeding

        print(f"[INFO] Produced {len(self.q4nx_tensors)} Q4NX vision tensors")
        self._export_weights(q4nx_path, "vision")


class Qwen35_2B(Qwen35, model_arch=ModelArch.QWEN35_2B):
    print("[INFO] Using Qwen35_2B converter")
    pass


class Qwen35_08B(Qwen35, model_arch=ModelArch.QWEN35_08B):
    print("[INFO] Using Qwen35_08B converter")
    pass

class Qwen35_9B(Qwen35, model_arch=ModelArch.QWEN35_9B):
    print("[INFO] Using Qwen35_9B converter")
    pass
