pub mod decoder;
pub mod encoder;

use burn::{
    module::Module,
    record::{FullPrecisionSettings, Recorder},
    tensor::{backend::Backend, Device},
};
use burn_import::safetensors::SafetensorsFileRecorder;

use decoder::RNNTHead;
use encoder::ConformerEncoder;

// Default model hyperparameters for v3_e2e_rnnt
pub const FEAT_IN: usize = 64;
pub const N_LAYERS: usize = 16;
pub const D_MODEL: usize = 768;
pub const SUBSAMPLING_FACTOR: usize = 4;
pub const FF_EXPANSION_FACTOR: usize = 4;
pub const N_HEADS: usize = 16;
pub const CONV_KERNEL_SIZE: usize = 5;
pub const SUBS_KERNEL_SIZE: usize = 5;
pub const POS_EMB_MAX_LEN: usize = 5000;
pub const NUM_CLASSES: usize = 1025;
pub const PRED_HIDDEN: usize = 320;
pub const JOINT_HIDDEN: usize = 320;

#[derive(Module, Debug)]
pub struct GigaAMASR<B: Backend> {
    pub encoder: ConformerEncoder<B>,
    pub head: RNNTHead<B>,
}

impl<B: Backend> GigaAMASR<B> {
    pub fn new(device: &Device<B>) -> Self {
        Self {
            encoder: ConformerEncoder::new(
                FEAT_IN,
                N_LAYERS,
                D_MODEL,
                SUBSAMPLING_FACTOR,
                FF_EXPANSION_FACTOR,
                N_HEADS,
                CONV_KERNEL_SIZE,
                SUBS_KERNEL_SIZE,
                POS_EMB_MAX_LEN,
                device,
            ),
            head: RNNTHead::new(NUM_CLASSES, PRED_HIDDEN, D_MODEL, JOINT_HIDDEN, device),
        }
    }

    pub fn load(weights_dir: &str, device: &Device<B>) -> Self {
        let mut model = Self::new(device);

        let encoder_path = format!("{}/v3_e2e_rnnt_encoder.safetensors", weights_dir);
        let head_path = format!("{}/v3_e2e_rnnt_head.safetensors", weights_dir);

        // Load encoder weights
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
        let encoder_record = recorder
            .load(encoder_path.into(), device)
            .expect("Failed to load encoder weights");
        model.encoder = model.encoder.load_record(encoder_record);

        // Load head weights with key remapping for LSTM and joint_net
        use burn_import::safetensors::LoadArgs;
        let head_args = LoadArgs::new(head_path.into())
            // LSTM: decoder.lstm.weight_ih_l0 -> decoder.lstm.weight_ih.weight
            .with_key_remap("decoder\\.lstm\\.weight_ih_l0", "decoder.lstm.weight_ih.weight")
            .with_key_remap("decoder\\.lstm\\.bias_ih_l0", "decoder.lstm.weight_ih.bias")
            .with_key_remap("decoder\\.lstm\\.weight_hh_l0", "decoder.lstm.weight_hh.weight")
            .with_key_remap("decoder\\.lstm\\.bias_hh_l0", "decoder.lstm.weight_hh.bias")
            // Joint: joint.joint_net.1.* -> joint.output.*
            .with_key_remap("joint\\.joint_net\\.1\\.(.+)", "joint.output.$1")
;
        let head_record = recorder
            .load(head_args, device)
            .expect("Failed to load head weights");
        model.head = model.head.load_record(head_record);

        model
    }
}
