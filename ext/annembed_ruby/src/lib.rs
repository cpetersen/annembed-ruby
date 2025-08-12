use magnus::{define_module, Error};

mod embedder;
mod svd;
mod utils;
mod clustering;

#[cfg(test)]
mod tests;

#[magnus::init]
fn init() -> Result<(), Error> {
    let module = define_module("AnnEmbed")?;
    
    // Initialize submodules
    embedder::init(&module)?;
    svd::init(&module)?;
    utils::init(&module)?;
    clustering::init(&module)?;
    
    Ok(())
}