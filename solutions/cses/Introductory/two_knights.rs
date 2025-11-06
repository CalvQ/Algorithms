use std::io;

fn main() {
    let mut k = String::new();
    io::stdin().read_line(&mut k).expect("Invalid input");
    let k: u128 = k.trim().parse().expect("Not an integer");

    println!("0");
    for i in 2..=k {
        // Generate number for ixi chessboard

        let total_configs: u128 = i * i * (i * i - 1);
        let collisions: u128 = match i {
            2 => 0,
            3 => 16,
            _ => {
                (4 * 12) // 4 corners
                + (4 * 10 * (i-4)) // 4 walls
                + (i-4) * (i-4) * 8 // center
            }
        };

        println!("{}", (total_configs - collisions) / 2)
    }
}
