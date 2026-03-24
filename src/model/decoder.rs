use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{activation, backend::Backend, Device, Int, Tensor},
};

// ---------------------------------------------------------------------------
// Manual LSTM cell matching PyTorch's weight layout
// Weights are loaded as Linear layers so the PyTorch adapter transposes them.
// PyTorch LSTM: gates = x @ W_ih^T + b_ih + h @ W_hh^T + b_hh
// With Linear in Burn (stores [in, out], forward does x @ W):
//   we need gates = linear_ih(x) + linear_hh(h)
//   where linear_ih.weight has shape [input_size, 4*hidden], bias [4*hidden]
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct LstmCell<B: Backend> {
    pub weight_ih: Linear<B>,
    pub weight_hh: Linear<B>,
    #[module(skip)]
    pub hidden_size: usize,
}

impl<B: Backend> LstmCell<B> {
    pub fn new(input_size: usize, hidden_size: usize, device: &Device<B>) -> Self {
        Self {
            weight_ih: LinearConfig::new(input_size, 4 * hidden_size).init(device),
            weight_hh: LinearConfig::new(hidden_size, 4 * hidden_size).init(device),
            hidden_size,
        }
    }

    #[allow(dead_code)]
    pub fn forward(
        &self,
        input: Tensor<B, 2>,
        h: Tensor<B, 2>,
        c: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let hs = self.hidden_size;
        let gates = self.weight_ih.forward(input) + self.weight_hh.forward(h);
        let b = gates.dims()[0];

        let i = activation::sigmoid(gates.clone().slice([0..b, 0..hs]));
        let f = activation::sigmoid(gates.clone().slice([0..b, hs..2 * hs]));
        let g = gates.clone().slice([0..b, 2 * hs..3 * hs]).tanh();
        let o = activation::sigmoid(gates.slice([0..b, 3 * hs..4 * hs]));

        let c_new = f * c + i * g;
        let h_new = o * c_new.clone().tanh();

        (h_new, c_new)
    }
}

// ---------------------------------------------------------------------------
// RNNTDecoder (Prediction Network)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct RNNTDecoder<B: Backend> {
    pub embed: Embedding<B>,
    pub lstm: LstmCell<B>,
}

impl<B: Backend> RNNTDecoder<B> {
    pub fn new(num_classes: usize, pred_hidden: usize, device: &Device<B>) -> Self {
        Self {
            embed: EmbeddingConfig::new(num_classes, pred_hidden).init(device),
            lstm: LstmCell::new(pred_hidden, pred_hidden, device),
        }
    }

    #[allow(dead_code)]
    pub fn predict(
        &self,
        x: Option<Tensor<B, 2, Int>>,
        h: Tensor<B, 2>,
        c: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let emb: Tensor<B, 3> = match x {
            Some(token_ids) => self.embed.forward(token_ids),
            None => {
                let device = h.device();
                Tensor::zeros([1, 1, self.lstm.hidden_size], &device)
            }
        };
        // emb: [batch, 1, pred_hidden] -> [batch, pred_hidden]
        let [b, _, d] = emb.dims();
        let emb: Tensor<B, 2> = emb.reshape([b, d]);
        let (h_new, c_new) = self.lstm.forward(emb, h, c);
        // g: [batch, pred_hidden]
        (h_new.clone(), h_new, c_new)
    }
}

// ---------------------------------------------------------------------------
// RNNTJoint
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct RNNTJoint<B: Backend> {
    pub enc: Linear<B>,
    pub pred: Linear<B>,
    pub output: Linear<B>,
}

impl<B: Backend> RNNTJoint<B> {
    pub fn new(
        enc_hidden: usize,
        pred_hidden: usize,
        joint_hidden: usize,
        num_classes: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            enc: LinearConfig::new(enc_hidden, joint_hidden).init(device),
            pred: LinearConfig::new(pred_hidden, joint_hidden).init(device),
            output: LinearConfig::new(joint_hidden, num_classes).init(device),
        }
    }

    #[allow(dead_code)]
    pub fn joint(&self, encoder_out: Tensor<B, 3>, decoder_out: Tensor<B, 3>) -> Tensor<B, 4> {
        let enc = self.enc.forward(encoder_out).unsqueeze_dim::<4>(2);
        let pred = self.pred.forward(decoder_out).unsqueeze_dim::<4>(1);
        let combined = activation::relu(enc + pred);
        let logits = self.output.forward(combined);
        activation::log_softmax(logits, 3)
    }
}

// ---------------------------------------------------------------------------
// RNNTHead
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct RNNTHead<B: Backend> {
    pub decoder: RNNTDecoder<B>,
    pub joint: RNNTJoint<B>,
}

impl<B: Backend> RNNTHead<B> {
    pub fn new(
        num_classes: usize,
        pred_hidden: usize,
        enc_hidden: usize,
        joint_hidden: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            decoder: RNNTDecoder::new(num_classes, pred_hidden, device),
            joint: RNNTJoint::new(enc_hidden, pred_hidden, joint_hidden, num_classes, device),
        }
    }
}
