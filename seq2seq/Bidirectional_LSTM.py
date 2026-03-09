import torch
import torch.nn as nn

class SharedWeightBiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()
        self.input_size  = embedding_dim
        self.hidden_size = hidden_dim
        self.num_layers  = n_layers
        self.dropout     = dropout

        self.fwd_layers = nn.ModuleList()
        self.bwd_layers = nn.ModuleList()

        for layer in range(n_layers):
            layer_input_size = embedding_dim if layer == 0 else hidden_dim
            fwd_layer = self._make_lstm_layer(layer_input_size, hidden_dim)
            bwd_layer = self._make_lstm_layer(layer_input_size, hidden_dim,
                                              shared_layer=fwd_layer)
            self.fwd_layers.append(fwd_layer)
            self.bwd_layers.append(bwd_layer)

        self.dropout_layer = nn.Dropout(dropout)

    def _make_lstm_layer(self, input_size, hidden_size, shared_layer=None):
        if shared_layer is not None:
            return shared_layer
        layer = nn.ParameterDict({
            "W_ii": nn.Parameter(torch.randn(hidden_size, input_size)),
            "W_hi": nn.Parameter(torch.randn(hidden_size, hidden_size)),
            "b_i":  nn.Parameter(torch.zeros(hidden_size)),
            "W_if": nn.Parameter(torch.randn(hidden_size, input_size)),
            "W_hf": nn.Parameter(torch.randn(hidden_size, hidden_size)),
            "b_f":  nn.Parameter(torch.zeros(hidden_size)),
            "W_io": nn.Parameter(torch.randn(hidden_size, input_size)),
            "W_ho": nn.Parameter(torch.randn(hidden_size, hidden_size)),
            "b_o":  nn.Parameter(torch.zeros(hidden_size)),
            "W_ig": nn.Parameter(torch.randn(hidden_size, input_size)),
            "W_hg": nn.Parameter(torch.randn(hidden_size, hidden_size)),
            "b_g":  nn.Parameter(torch.zeros(hidden_size)),
        })
        return layer

    def _lstm_cell(self, x_t, h_t, c_t, params):
        i_t = torch.sigmoid(x_t @ params["W_ii"].T + h_t @ params["W_hi"].T + params["b_i"])
        f_t = torch.sigmoid(x_t @ params["W_if"].T + h_t @ params["W_hf"].T + params["b_f"])
        o_t = torch.sigmoid(x_t @ params["W_io"].T + h_t @ params["W_ho"].T + params["b_o"])
        g_t = torch.tanh(   x_t @ params["W_ig"].T + h_t @ params["W_hg"].T + params["b_g"])
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def forward(self, x, state=None):
        seq_len, batch, _ = x.size()
        if state is None:
            h_f = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
            c_f = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
            h_b = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
            c_b = torch.zeros(self.num_layers, batch, self.hidden_size, device=x.device)
        else:
            (h_f, c_f), (h_b, c_b) = state

        layer_input = x
        for layer_idx in range(self.num_layers):
            fwd_params = self.fwd_layers[layer_idx]
            bwd_params = self.bwd_layers[layer_idx]
            h_t_f = h_f[layer_idx].clone()
            c_t_f = c_f[layer_idx].clone()
            h_t_b = h_b[layer_idx].clone()
            c_t_b = c_b[layer_idx].clone()

            fwd_outputs = []
            for t in range(seq_len):
                h_t_f, c_t_f = self._lstm_cell(layer_input[t], h_t_f, c_t_f, fwd_params)
                fwd_outputs.append(h_t_f.unsqueeze(0))

            bwd_outputs = [None] * seq_len
            for t in reversed(range(seq_len)):
                h_t_b, c_t_b = self._lstm_cell(layer_input[t], h_t_b, c_t_b, bwd_params)
                bwd_outputs[t] = h_t_b.unsqueeze(0)

            fwd = torch.cat(fwd_outputs, dim=0)
            bwd = torch.cat(bwd_outputs, dim=0)
            layer_output = fwd + bwd

            h_f[layer_idx] = h_t_f
            c_f[layer_idx] = c_t_f
            h_b[layer_idx] = h_t_b
            c_b[layer_idx] = c_t_b

            if layer_idx < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
            layer_input = layer_output

        return layer_input, ((h_f, c_f), (h_b, c_b))
