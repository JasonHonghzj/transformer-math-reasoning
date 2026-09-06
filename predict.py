import sys
import math
import torch


device = torch.device("cpu")


characters = "()+0123456789="
TOKENS = ["<bos>", "<eos>", "<pad>"] + [c for c in characters]

TOKEN_MAP = dict((t, i) for i, t in enumerate(TOKENS))

BOS = TOKEN_MAP["<bos>"]
EOS = TOKEN_MAP["<eos>"]
PAD = TOKEN_MAP["<pad>"]


def encode(s, *, eos = False):
    """
    transformed input string into token id：
    [BOS, <chars...>, (optional EOS)]
    """
    if s.startswith("<bos>"):
        s = s[5:]

    output = [BOS]

    for c in s:
        if c ==" ":
            continue
            
        output.append(TOKEN_MAP[c])

    if eos:
        output.append(EOS)

    return torch.tensor(output, device=device, dtype=torch.long)


def decode(token_ids):
    """
    turn token_id back into string and remove <bos> <eos> <pad>
    """
    chars = []
    for idx in token_ids:
        tok = TOKENS[int(idx)]
        if tok in ("<bos>", "<eos>", "<pad>"):
            continue
        chars.append(tok)
    return "".join(chars)


def causal_mask(T):
    m = torch.full((T, T), float("-inf"))
    m = torch.triu(m, diagonal=1)
    return m



class MathTransformer(torch.nn.Module):
    def __init__(self, d_model=192, nhead=6, num_layers=6, dim_ff=384, max_len=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        vocab_size = len(TOKENS)

        # token + position embeddings
        self.tok_emb = torch.nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = torch.nn.Embedding(max_len, d_model)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.blocks = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = torch.nn.Linear(d_model, vocab_size)

        # init
        torch.nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.lm_head.bias)

    def forward(self, x):
        # x: (N, T)
        N, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(pos)  # (N, T, d_model)

        # key padding mask: True where we want to ignore (PAD)
        key_padding_mask = (x == PAD)  # (N, T) bool

        # causal mask for self-attention (float, -inf above diagonal)
        attn_mask = causal_mask(T).to(x.device)  # (T, T)

        h = self.blocks(
            h,
            mask=attn_mask,                        # causal
            src_key_padding_mask=key_padding_mask  # pad masking
        )
        logits = self.lm_head(h)  # (N, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, prefix_ids, max_new_tokens=128):
        self.eval()
        x = prefix_ids.clone().to(next(self.parameters()).device)  # (N, T0)
        for _ in range(max_new_tokens):
            if x.size(1) >= self.max_len:
                break
            logits = self.forward(x)[:, -1, :]   # (N, V)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)  # greedy: (N, 1)
            x = torch.cat([x, next_id], dim=1)
            if (next_id == EOS).all():
                break
        return x



# --------------------------------------------------
def main():
    if len(sys.argv) != 2:
        sys.exit(1)

    input_filename = sys.argv[1]

    # Model building 
    model = MathTransformer(
        d_model=192,
        nhead=6,
        num_layers=6,
        dim_ff=384,
        max_len=128,
        dropout=0.1,
    ).to(device)

    state_dict = torch.load("math.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    
    with open(input_filename, "r") as f:
        for line in f:
            prompt = line.strip()
            if prompt == "":
                print("")
                continue


            prefix = encode(prompt, eos=False).unsqueeze(0)  # (1, T)

            with torch.no_grad():
                full_ids = model.generate(prefix, max_new_tokens=128)[0].tolist()

            output_str = decode(full_ids)
            print(output_str)


if __name__ == "__main__":
    main()
