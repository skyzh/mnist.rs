#[macro_use]
extern crate rulinalg;

mod network;
use self::network::Network;

fn main() {
    let network = Network::new_mnist();
}
