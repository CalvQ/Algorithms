use std::io;

fn main() {
    let mut k = String::new();
    io::stdin().read_line(&mut k).expect("Invalid input");
    let k: u128 = k.trim().parse().expect("Not an integer");

    let mut set1 = Vec::new();
    let mut set2 = Vec::new();
    let mut sum1: u128 = 0;
    let mut sum2: u128 = 0;

    for i in (1..=k).rev() {
        if sum1 > sum2 {
            //append to 2nd set
            sum2 += i;
            set2.push(i);
        } else {
            //append to 1st set
            sum1 += i;
            set1.push(i);
        }
    }

    // print output
    if sum1 != sum2 {
        println!("NO");
        return;
    }

    println!("YES");
    println!("{}", set1.len());
    for item in &set1 {
        print!("{} ", item);
    }
    println!();
    println!("{}", set2.len());
    for item in &set2 {
        print!("{} ", item);
    }
    println!();
}
