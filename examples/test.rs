fn main() {
    let arr = [7, 4, 1, 9, 0];
    let c = &arr[..3];
    for i in c {
        println!("{i}")
    }
}
