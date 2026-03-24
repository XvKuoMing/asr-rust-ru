use burn::{
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d,
    },
    tensor::{activation, backend::Backend, Bool, Device, Int, Tensor},
};

// ---------------------------------------------------------------------------
// Rotary Positional Embedding (no trainable weights)
// ---------------------------------------------------------------------------

pub fn build_rotary_embedding<B: Backend>(
    length: usize,
    dim: usize,
    base: f32,
    device: &Device<B>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f32 / dim as f32))
        .collect();
    let inv = Tensor::<B, 1>::from_floats(inv_freq.as_slice(), device);
    let t: Vec<f32> = (0..length).map(|i| i as f32).collect();
    let t = Tensor::<B, 1>::from_floats(t.as_slice(), device);

    // freqs: [length, half]
    let freqs: Tensor<B, 2> = t.unsqueeze_dim(1).matmul(inv.unsqueeze_dim(0));
    // emb: [length, dim]
    let emb: Tensor<B, 2> = Tensor::cat(vec![freqs.clone(), freqs], 1);
    let cos: Tensor<B, 4> = emb.clone().cos().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
    let sin: Tensor<B, 4> = emb.sin().unsqueeze_dim::<3>(1).unsqueeze_dim::<4>(2);
    (cos, sin)
}

fn rtt_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = x.dims();
    let half = dims[3] / 2;
    let x1 = x.clone().slice([0..dims[0], 0..dims[1], 0..dims[2], 0..half]);
    let x2 = x.slice([0..dims[0], 0..dims[1], 0..dims[2], half..dims[3]]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

pub fn apply_rotary_pos_emb<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let t = q.dims()[0];
    let d = cos.dims()[3];
    let cos = cos.slice([0..t, 0..1, 0..1, 0..d]);
    let d = sin.dims()[3];
    let sin = sin.slice([0..t, 0..1, 0..1, 0..d]);
    let q_rot = q.clone() * cos.clone() + rtt_half(q) * sin.clone();
    let k_rot = k.clone() * cos + rtt_half(k) * sin;
    (q_rot, k_rot)
}

// ---------------------------------------------------------------------------
// StridingSubsampling (conv1d variant)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct StridingSubsampling<B: Backend> {
    conv: Vec<Conv1d<B>>,
    #[module(skip)]
    kernel_size: usize,
}

impl<B: Backend> StridingSubsampling<B> {
    pub fn new(
        feat_in: usize,
        feat_out: usize,
        kernel_size: usize,
        subsampling_factor: usize,
        device: &Device<B>,
    ) -> Self {
        let n_convs = (subsampling_factor as f64).log2() as usize;
        let stride = 2;
        let padding = (kernel_size - 1) / 2;
        let mut convs = Vec::new();
        let mut in_ch = feat_in;
        for _ in 0..n_convs {
            convs.push(
                Conv1dConfig::new(in_ch, feat_out, kernel_size)
                    .with_stride(stride)
                    .with_padding(PaddingConfig1d::Explicit(padding))
                    .init(device),
            );
            in_ch = feat_out;
        }
        Self { conv: convs, kernel_size }
    }

    pub fn forward(&self, x: Tensor<B, 3>, lengths: Tensor<B, 1>) -> (Tensor<B, 3>, Tensor<B, 1>) {
        // x: [B, T, feat_in] -> transpose for conv1d -> [B, feat_in, T]
        let mut out = x.swap_dims(1, 2);
        for conv in &self.conv {
            out = activation::relu(conv.forward(out));
        }
        // out is [B, d_model, T']
        let out = out.swap_dims(1, 2); // [B, T', d_model]
        let new_lengths = self.calc_output_length(lengths);
        (out, new_lengths)
    }

    fn calc_output_length(&self, lengths: Tensor<B, 1>) -> Tensor<B, 1> {
        let n_convs = self.conv.len();
        let kernel_size = self.kernel_size;
        let padding = (kernel_size - 1) / 2;
        let add_pad = (2 * padding) as f32 - kernel_size as f32;

        let mut l = lengths;
        for _ in 0..n_convs {
            l = ((l + add_pad) / 2.0 + 1.0).floor();
        }
        l
    }
}

// ---------------------------------------------------------------------------
// ConformerFeedForward
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConformerFeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> ConformerFeedForward<B> {
    pub fn new(d_model: usize, d_ff: usize, device: &Device<B>) -> Self {
        Self {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear2
            .forward(activation::silu(self.linear1.forward(x)))
    }
}

// ---------------------------------------------------------------------------
// ConformerConvolution
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConformerConvolution<B: Backend> {
    pointwise_conv1: Conv1d<B>,
    depthwise_conv: Conv1d<B>,
    batch_norm: LayerNorm<B>,
    pointwise_conv2: Conv1d<B>,
}

impl<B: Backend> ConformerConvolution<B> {
    pub fn new(d_model: usize, kernel_size: usize, device: &Device<B>) -> Self {
        let padding = (kernel_size - 1) / 2;
        Self {
            pointwise_conv1: Conv1dConfig::new(d_model, d_model * 2, 1).init(device),
            depthwise_conv: Conv1dConfig::new(d_model, d_model, kernel_size)
                .with_padding(PaddingConfig1d::Explicit(padding))
                .with_groups(d_model)
                .init(device),
            batch_norm: LayerNormConfig::new(d_model).init(device),
            pointwise_conv2: Conv1dConfig::new(d_model, d_model, 1).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, pad_mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        // x: [B, T, d_model]
        let x = x.swap_dims(1, 2); // [B, d_model, T]
        let x = self.pointwise_conv1.forward(x);
        let x = glu(x, 1);
        let x = match pad_mask {
            Some(ref mask) => {
                let mask3: Tensor<B, 3, Bool> = mask.clone().unsqueeze_dim(1);
                x.mask_fill(mask3, 0.0f32)
            }
            None => x,
        };
        let x = self.depthwise_conv.forward(x);
        // LayerNorm expects [B, T, C]
        let x = x.swap_dims(1, 2);
        let x = self.batch_norm.forward(x);
        let x = x.swap_dims(1, 2);
        let x = activation::silu(x);
        let x = self.pointwise_conv2.forward(x);
        x.swap_dims(1, 2) // [B, T, d_model]
    }
}

fn glu<B: Backend>(x: Tensor<B, 3>, dim: usize) -> Tensor<B, 3> {
    let size = x.dims()[dim];
    let half = size / 2;
    let ranges: Vec<std::ops::Range<usize>> = (0..3)
        .map(|i| {
            if i == dim {
                0..half
            } else {
                0..x.dims()[i]
            }
        })
        .collect();
    let ranges2: Vec<std::ops::Range<usize>> = (0..3)
        .map(|i| {
            if i == dim {
                half..size
            } else {
                0..x.dims()[i]
            }
        })
        .collect();
    let a = x.clone().slice([ranges[0].clone(), ranges[1].clone(), ranges[2].clone()]);
    let b = x.slice([ranges2[0].clone(), ranges2[1].clone(), ranges2[2].clone()]);
    a * activation::sigmoid(b)
}

// ---------------------------------------------------------------------------
// RotaryPositionMultiHeadAttention
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct RotaryPositionMultiHeadAttention<B: Backend> {
    linear_q: Linear<B>,
    linear_k: Linear<B>,
    linear_v: Linear<B>,
    linear_out: Linear<B>,
    #[module(skip)]
    n_head: usize,
    #[module(skip)]
    d_k: usize,
}

impl<B: Backend> RotaryPositionMultiHeadAttention<B> {
    pub fn new(n_head: usize, n_feat: usize, device: &Device<B>) -> Self {
        let d_k = n_feat / n_head;
        Self {
            linear_q: LinearConfig::new(n_feat, n_feat).init(device),
            linear_k: LinearConfig::new(n_feat, n_feat).init(device),
            linear_v: LinearConfig::new(n_feat, n_feat).init(device),
            linear_out: LinearConfig::new(n_feat, n_feat).init(device),
            n_head,
            d_k,
        }
    }

    pub fn forward(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        att_mask: Option<Tensor<B, 3, Bool>>,
    ) -> Tensor<B, 3> {
        let [b, t, _] = query.dims();
        let h = self.n_head;
        let d_k = self.d_k;

        // Apply RoPE before projecting through Q/K
        // Python does: query.transpose(0,1).view(t, b, h, d_k) then RoPE then project
        // We replicate the same logic
        let q_pre = query.clone().swap_dims(0, 1).reshape([t, b, h, d_k]);
        let k_pre = key.clone().swap_dims(0, 1).reshape([t, b, h, d_k]);
        let v_pre = value.clone().swap_dims(0, 1).reshape([t, b, h, d_k]);

        let (q_rot, k_rot) = apply_rotary_pos_emb(q_pre, k_pre, cos, sin);

        let q_lin = q_rot.reshape([t, b, h * d_k]).swap_dims(0, 1); // [B, T, n_feat]
        let k_lin = k_rot.reshape([t, b, h * d_k]).swap_dims(0, 1);
        let v_lin = v_pre.reshape([t, b, h * d_k]).swap_dims(0, 1);

        let q = self.linear_q.forward(q_lin).reshape([b, t, h, d_k]).swap_dims(1, 2);
        let k = self.linear_k.forward(k_lin).reshape([b, t, h, d_k]).swap_dims(1, 2);
        let v = self.linear_v.forward(v_lin).reshape([b, t, h, d_k]).swap_dims(1, 2);

        // Scaled dot-product attention
        #[cfg(feature = "flash-attn")]
        let attn_out = {
            let q_fa = q.swap_dims(1, 2); // [B, T, h, d_k]
            let k_fa = k.swap_dims(1, 2);
            let v_fa = v.swap_dims(1, 2);
            let out = burn_attention::FlashAttentionV3::forward(q_fa, k_fa, v_fa, None, false);
            out.reshape([b, t, h * d_k])
        };

        #[cfg(not(feature = "flash-attn"))]
        let attn_out = {
            let scale = 1.0 / (d_k as f64).sqrt();
            let scores = q.matmul(k.swap_dims(2, 3)) * scale;
            let scores = match att_mask {
                Some(mask) => {
                    let mask: Tensor<B, 4, Bool> = mask.unsqueeze_dim(1);
                    scores.mask_fill(mask, -10000.0f32)
                }
                None => scores,
            };
            let attn = activation::softmax(scores, 3);
            let out = attn.matmul(v);
            out.swap_dims(1, 2).reshape([b, t, h * d_k])
        };

        self.linear_out.forward(attn_out)
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConformerLayer<B: Backend> {
    norm_feed_forward1: LayerNorm<B>,
    feed_forward1: ConformerFeedForward<B>,
    norm_self_att: LayerNorm<B>,
    self_attn: RotaryPositionMultiHeadAttention<B>,
    norm_conv: LayerNorm<B>,
    conv: ConformerConvolution<B>,
    norm_feed_forward2: LayerNorm<B>,
    feed_forward2: ConformerFeedForward<B>,
    norm_out: LayerNorm<B>,
}

impl<B: Backend> ConformerLayer<B> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            norm_feed_forward1: LayerNormConfig::new(d_model).init(device),
            feed_forward1: ConformerFeedForward::new(d_model, d_ff, device),
            norm_self_att: LayerNormConfig::new(d_model).init(device),
            self_attn: RotaryPositionMultiHeadAttention::new(n_heads, d_model, device),
            norm_conv: LayerNormConfig::new(d_model).init(device),
            conv: ConformerConvolution::new(d_model, conv_kernel_size, device),
            norm_feed_forward2: LayerNormConfig::new(d_model).init(device),
            feed_forward2: ConformerFeedForward::new(d_model, d_ff, device),
            norm_out: LayerNormConfig::new(d_model).init(device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
        att_mask: Option<Tensor<B, 3, Bool>>,
        pad_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let fc_factor = 0.5;

        // FF1
        let residual = x;
        let x = self.norm_feed_forward1.forward(residual.clone());
        let x = self.feed_forward1.forward(x);
        let residual = residual + x * fc_factor;

        // Self attention
        let x = self.norm_self_att.forward(residual.clone());
        let x = self.self_attn.forward(
            x.clone(),
            x.clone(),
            x,
            cos,
            sin,
            att_mask,
        );
        let residual = residual + x;

        // Convolution
        let x = self.norm_conv.forward(residual.clone());
        let x = self.conv.forward(x, pad_mask);
        let residual = residual + x;

        // FF2
        let x = self.norm_feed_forward2.forward(residual.clone());
        let x = self.feed_forward2.forward(x);
        let residual = residual + x * fc_factor;

        self.norm_out.forward(residual)
    }
}

// ---------------------------------------------------------------------------
// ConformerEncoder (top-level)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
pub struct ConformerEncoder<B: Backend> {
    pre_encode: StridingSubsampling<B>,
    layers: Vec<ConformerLayer<B>>,
    #[module(skip)]
    pub n_heads: usize,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pos_emb_max_len: usize,
}

impl<B: Backend> ConformerEncoder<B> {
    pub fn new(
        feat_in: usize,
        n_layers: usize,
        d_model: usize,
        subsampling_factor: usize,
        ff_expansion_factor: usize,
        n_heads: usize,
        conv_kernel_size: usize,
        subs_kernel_size: usize,
        pos_emb_max_len: usize,
        device: &Device<B>,
    ) -> Self {
        let pre_encode = StridingSubsampling::new(
            feat_in,
            d_model,
            subs_kernel_size,
            subsampling_factor,
            device,
        );
        let d_ff = d_model * ff_expansion_factor;
        let layers = (0..n_layers)
            .map(|_| ConformerLayer::new(d_model, d_ff, n_heads, conv_kernel_size, device))
            .collect();

        Self {
            pre_encode,
            layers,
            n_heads,
            d_model,
            pos_emb_max_len,
        }
    }

    pub fn forward(
        &self,
        audio_signal: Tensor<B, 3>,
        length: Tensor<B, 1>,
    ) -> (Tensor<B, 3>, Tensor<B, 1>) {
        let device = audio_signal.device();

        // Subsampling: input [B, feat_in, T] transposed to [B, T, feat_in] inside pre_encode
        let (audio_signal, length) = self.pre_encode.forward(
            audio_signal.swap_dims(1, 2), // Python does audio_signal.transpose(1,2) before pre_encode
            length,
        );
        // audio_signal: [B, T', d_model], length: [B]

        let max_len = audio_signal.dims()[1];
        let d_k = self.d_model / self.n_heads;
        let (cos, sin) = build_rotary_embedding(max_len, d_k, 10000.0, &device);

        // Build masks
        let batch_size = audio_signal.dims()[0];
        let arange: Tensor<B, 2, Int> = Tensor::<B, 1, Int>::arange(0..max_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .expand([batch_size, max_len]);
        let len_int: Tensor<B, 2, Int> = length.clone().int().unsqueeze_dim::<2>(1).expand([batch_size, max_len]);
        let pad_mask: Tensor<B, 2, Bool> = arange.lower(len_int); // true where valid

        let att_mask: Option<Tensor<B, 3, Bool>> = if batch_size > 1 {
            let m: Tensor<B, 3, Int> = pad_mask.clone().int().unsqueeze_dim::<3>(1).expand([batch_size, max_len, max_len]);
            let m_t: Tensor<B, 3, Int> = pad_mask.clone().int().unsqueeze_dim::<3>(2).expand([batch_size, max_len, max_len]);
            let combined: Tensor<B, 3, Bool> = (m * m_t).bool();
            Some(combined.bool_not())
        } else {
            None
        };

        let pad_mask_inv: Tensor<B, 2, Bool> = pad_mask.bool_not();

        // Run conformer layers
        let mut x = audio_signal;
        for layer in &self.layers {
            x = layer.forward(
                x,
                cos.clone(),
                sin.clone(),
                att_mask.clone(),
                Some(pad_mask_inv.clone()),
            );
        }

        // Output: [B, d_model, T']
        (x.swap_dims(1, 2), length)
    }
}
