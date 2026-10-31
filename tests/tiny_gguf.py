"""Build a tiny random-weight llama GGUF (byte-level SPM vocab + ChatML template) for tests."""
import sys
import numpy as np
import gguf

out = sys.argv[1]
n_embd, n_layer, n_head, n_ff = 64, 2, 4, 128
tokens, scores, types = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>"], [0.0] * 5, [2, 3, 3, 3, 3]
for b in range(256):
    tokens.append(f"<0x{b:02X}>"); scores.append(0.0); types.append(6)
for w in ["▁", "a", "e", "o", "▁the", "▁a"]:
    tokens.append(w); scores.append(-1.0); types.append(1)
n_vocab = len(tokens)
w = gguf.GGUFWriter(out, "llama")
w.add_name("nexora-test-tiny")
w.add_context_length(512)
w.add_embedding_length(n_embd)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head)
w.add_rope_dimension_count(n_embd // n_head)
w.add_layer_norm_rms_eps(1e-5)
w.add_file_type(gguf.LlamaFileType.ALL_F32)
w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(4)
w.add_unk_token_id(0)
w.add_add_bos_token(False)
w.add_chat_template("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
rng = np.random.default_rng(0)
R = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
w.add_tensor("token_embd.weight", R(n_vocab, n_embd))
for i in range(n_layer):
    p = f"blk.{i}."
    w.add_tensor(p + "attn_norm.weight", np.ones(n_embd, np.float32))
    for t in ("attn_q", "attn_k", "attn_v", "attn_output"):
        w.add_tensor(p + t + ".weight", R(n_embd, n_embd))
    w.add_tensor(p + "ffn_norm.weight", np.ones(n_embd, np.float32))
    w.add_tensor(p + "ffn_gate.weight", R(n_ff, n_embd))
    w.add_tensor(p + "ffn_up.weight", R(n_ff, n_embd))
    w.add_tensor(p + "ffn_down.weight", R(n_embd, n_ff))
w.add_tensor("output_norm.weight", np.ones(n_embd, np.float32))
w.add_tensor("output.weight", R(n_vocab, n_embd))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("ok", out)
