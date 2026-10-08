// use anyhow::Ok;
use crate::error::Result;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use crate::agent::Agent;

pub async fn run_repl(agent: &mut Agent) -> Result<()> {
    let mut editor = DefaultEditor::new()?;

    loop {
        match editor.readline("\n > ") {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
            }
            Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                println!("\n 再见")
            }
            Err(e) => return Err(e.into()),
        }
    }

    Ok(())
}
