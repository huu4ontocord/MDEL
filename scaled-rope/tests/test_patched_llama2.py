import torch
from flash_patch import patch_model
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def test_flash_attention_patch(dtype=torch.float16, device="cuda:0"):
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    config.max_position_embeddings = 8192
    interpolation_factor = config.max_position_embeddings / 2048

    config.rope_scaling = {"type": "linear", "factor": interpolation_factor}

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", use_auth_token=True
    )
    tokenizer.add_special_tokens(
        {"pad_token": "</s>", "eos_token": "</s>", "sep_token": "<s>"}
    )

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=dtype
    ).to(device)
    device2 = "cuda:1"
    patched_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=dtype
    ).to(device2)
    patch_model(patched_model, resid_pdrop=None, flash_attention=True)

    device = model.device
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    with torch.no_grad():
        for layer1, layer2 in zip(model.model.layers, patched_model.model.layers):
            hidden_states = torch.randn(
                4, 10, head_dim * n_heads, dtype=dtype, device=device
            )
            attention_mask = (
                torch.randn(4, 10, device=device).sort(dim=-1).values < 0.5
            ).int()
            attn_mask = patched_model.model._prepare_decoder_attention_mask(
                attention_mask, (4, 10), hidden_states, 0
            )
            position_ids = torch.arange(10, device=device).unsqueeze(0).expand(4, -1)
            attn1, attn2 = layer1.self_attn, layer2.self_attn

            out1, _, _ = attn1(
                hidden_states, attention_mask=attn_mask, position_ids=position_ids
            )
            out2, _, _ = attn2(
                hidden_states.to(device2),
                attention_mask=attn_mask.to(device2),
                position_ids=position_ids.to(device2),
            )

            assert (
                ((out1 - out2.to(device)) * attention_mask.unsqueeze(-1))
                .mean(dim=-1)
                .abs()
                < 1e-3
            ).all()

        batch = tokenizer(
            ["hello world", "lorem ipsum dolor sit amet"],
            padding=True,
            return_tensors="pt",
        ).to(device)
        out1 = model(**batch).logits
        batch = batch.to(device2)
        out2 = patched_model(**batch).logits

        diff = (out1.to(device2) - out2) * batch["attention_mask"].unsqueeze(-1)
        assert (diff.abs() < 0.1).all()

    input_ids = torch.randint(0, model.config.vocab_size, size=(2, 10), device=device2)
    patched_model(input_ids).logits.mean().backward()


if __name__ == "__main__":
    test_flash_attention_patch()
