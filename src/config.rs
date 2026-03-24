use envconfig::Envconfig;

#[derive(Envconfig, Clone, Debug)]
pub struct AppConfig {
    #[envconfig(from = "HOST", default = "0.0.0.0")]
    pub host: String,

    #[envconfig(from = "PORT", default = "8080")]
    pub port: u16,

    #[envconfig(from = "BATCH_SIZE", default = "16")]
    pub batch_size: usize,

    #[envconfig(from = "BATCH_TIMEOUT_MS", default = "5")]
    pub batch_timeout_ms: u64,

    #[envconfig(from = "WEIGHTS_DIR", default = "weights")]
    pub weights_dir: String,
}
