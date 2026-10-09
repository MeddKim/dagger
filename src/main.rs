use clap::Parser;
use dagger::cli::{self, Cli};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("info,dagger=debug"))
        .init();

    let cli = Cli::parse();

    cli::run(cli).await
}
