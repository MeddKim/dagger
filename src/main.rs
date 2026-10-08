use clap::Parser;
use dagger::cli::{self, Cli};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    cli::run(cli).await
}
