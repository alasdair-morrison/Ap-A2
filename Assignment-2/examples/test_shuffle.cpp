#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

int main() {
    std::vector<int> v(10);
    std::iota(v.begin(), v.end(), 0);
    std::random_shuffle(v.begin(), v.end());
    for(int x : v) std::cout << x << " ";
    return 0;
}
