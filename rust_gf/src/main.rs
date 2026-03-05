mod config;
mod search;

use clap::Parser;
use regex::RegexBuilder;
use std::process;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Search pattern (regex)
    #[arg(short = 's', long = "search")]
    search: Option<String>,

    /// Target files/directories pattern
    #[arg(short = 't', long = "target")]
    target: Option<Vec<String>>,

    /// Ignore files/directories pattern
    #[arg(short = 'i', long = "ignore")]
    ignore: Option<Vec<String>>,

    /// Context lines to show
    #[arg(short = 'c', long, default_value_t = 0)]
    context: usize,

    /// Maximum lines to read per file
    #[arg(short = 'm', long = "maxline")]
    maxline: Option<usize>,

    /// Enable debug output
    #[arg(long)]
    debug: bool,
}

fn main() {
    let args = Args::parse();

    // 1. Load Config
    let mut config = config::Config::new();

    // 2. Merge Args into Config strings
    if let Some(targets) = args.target {
        for t in targets {
            for part in t.split(',') {
                if !part.trim().is_empty() {
                    config.target_strings.insert(part.trim().to_string());
                }
            }
        }
    }

    if let Some(ignores) = args.ignore {
        for i in ignores {
            for part in i.split(',') {
                if !part.trim().is_empty() {
                    config.ignore_strings.insert(part.trim().to_string());
                }
            }
        }
    }

    // 3. Compile all config regexes
    config.compile();

    // 4. Search Term
    let search_term = match args.search {
        Some(term) => term,
        None => {
            eprintln!("Error: No search pattern provided.");
            process::exit(1);
        }
    };

    if args.debug {
        println!("Search Term: {}", search_term);
        println!("Ignore patterns: {:?}", config.ignore_strings);
        println!("Target patterns: {:?}", config.target_strings);
    }

    // 5. Compile search regex
    let pattern = match RegexBuilder::new(&search_term).case_insensitive(true).build() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error compiling regex '{}': {}", search_term, e);
            process::exit(1);
        }
    };

    // 6. Run Search
    let options = search::SearchOptions {
        pattern,
        context: args.context,
        max_line: args.maxline,
        debug: args.debug,
    };

    search::recursive_search(&config, &options);
}
