// Research accelerator for the existing pile-orbit and strip-gap-orbit transfers.
// Sources: synthesized/matrixMeanders/matrixAxis{Pile,StripGap}Orbit.py.
// Build: g++ -O3 -march=native -std=c++20 -fopenmp research/matrixMeanders/axis_transfer.cpp -o /tmp/axis_transfer
// Run: OMP_NUM_THREADS=8 /tmp/axis_transfer pile-wide 32 4
// Counts use checked uint64_t arithmetic. Keys use 64 bits (pile) or 128 (gap).
// Final layers are streamed in parallel instead of storing their enormous maps.
// The full doubled sum is bounded by 2*4^31 for supported indices, including
// all intermediate products and parallel partial sums; these fit uint64_t.
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

using U64 = uint64_t;
using U128 = unsigned __int128;

int width(U64 x) { return std::bit_width(x); }
int width(U128 x) {
    U64 high = x >> 64;
    return high ? 64 + width(high) : width(U64(x));
}
U64 add_checked(U64 a, U64 b) {
    if (b > std::numeric_limits<U64>::max() - a) throw std::overflow_error("count overflow");
    return a + b;
}
U64 mix(U64 x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
template<class K> U64 hash_key(K key) {
    if constexpr (sizeof(K) > 8) return mix(U64(key)) ^ mix(U64(key >> 64));
    else return mix(key);
}

// Zero is never a transfer key. Keep a bounded-load table without node allocations.
template<class K> class Counts {
public:
    struct Slot { K key = 0; U64 count = 0; };
    std::vector<Slot> slots;
    size_t size = 0;
    explicit Counts(size_t capacity = 16): slots(capacity) {}
    void insert(K key, U64 count) {
        if (!key) throw std::runtime_error("zero transfer key");
        if ((size + 1) * 10 > slots.size() * 7) {
            Counts larger(slots.size() * 2);
            for (const auto &s: slots) if (s.key) larger.insert(s.key, s.count);
            *this = std::move(larger);
        }
        size_t i = hash_key(key) & (slots.size() - 1);
        while (slots[i].key && slots[i].key != key) i = (i + 1) & (slots.size() - 1);
        if (!slots[i].key) { slots[i].key = key; ++size; }
        slots[i].count = add_checked(slots[i].count, count);
    }
};

bool joined(const std::vector<int> &partner) {
    if (partner.empty()) throw std::runtime_error("empty partner lookup");
    int terminal = partner[0], strands = 1, size = partner.size();
    while (terminal != size / 2) {
        if (terminal <= 0 || terminal >= size || strands > size)
            throw std::runtime_error("invalid partner cycle");
        terminal = partner[size - terminal]; ++strands;
    }
    return 2 * strands == size;
}

U64 locator(U64) { return std::numeric_limits<U64>::max() / 3; }
U64 reflect_pile(U64 c) {
    U64 mask = locator(c);
    return ((c & mask) << 1) | ((c >> 1) & mask);
}
template<class Word> bool half_pile_bits(Word c) {
    Word mask = ~Word(0) / 3, a = c & mask, z = (c >> 1) & mask;
    if (width(a) < width(z) || width(a) > width(z) + 4) return false;
    int size = (width(a) + width(z)) / 2 - 1;
    int partner[128], left[128], right[128];
    for (int i = 0; i < size; ++i) partner[i] = i;
    auto pair = [&](Word side, int terminal, int step, int *cross) {
        int open[128], opened = 0, crossed = 0;
        while (side > 1) {
            if (!(side & 1)) open[opened++] = terminal;
            else if (opened) {
                int other = open[--opened];
                partner[terminal] = other; partner[other] = terminal;
            } else cross[crossed++] = terminal;
            terminal += step; side >>= 2;
        }
        return crossed;
    };
    int nl = pair(a, 0, 1, left), nr = pair(z, size - 1, -1, right);
    if (nl != nr) throw std::runtime_error("unpaired pile terminals");
    for (int i = 0; i < nl; ++i) {
        partner[left[i]] = right[i]; partner[right[i]] = left[i];
    }
    int terminal = partner[0], strands = 1;
    while (terminal != size / 2) {
        terminal = partner[size - terminal]; ++strands;
    }
    return 2 * strands == size;
}
bool half_pile(U64 c) { return half_pile_bits(c); }
U64 walk_pile(U64 x) {
    int balance = 0; U64 bit = 1;
    while (balance >= 0) {
        bit <<= 2;
        if (!bit) throw std::overflow_error("pile walk overflow");
        balance += (x & bit) ? -1 : 1;
    }
    return bit;
}
template<class Emit> void pile_transitions(U64 c, Emit emit) {
    U64 mask = locator(c), a = c & mask, z = (c >> 1) & mask;
    U64 ae = (a & 1) ^ 1, ze = (z & 1) ^ 1;
    emit((c << 2) | 3);
    if (a > 1) emit((ae << 1) | (a >> 2) | (z << 3));
    if (z > 1) emit(ze | (a << 2) | (z >> 1));
    if (a > 1 && z > 1 && (ae || ze)) {
        if (ae && !ze) a ^= walk_pile(a);
        else if (ze && !ae) z ^= walk_pile(z);
        emit((((z >> 2) << 3) | a) >> 2);
    }
}

U128 reflect_pile_wide(U128 c) {
    U128 mask = ~U128(0) / 3;
    return ((c & mask) << 1) | ((c >> 1) & mask);
}
bool half_pile_wide(U128 c) {
    if (!(c >> 64)) return half_pile(U64(c));
    return half_pile_bits(c);
}
U128 walk_pile_wide(U128 x) {
    int balance = 0; U128 bit = 1;
    while (balance >= 0) {
        bit <<= 2;
        if (!bit) throw std::overflow_error("wide pile walk overflow");
        balance += (x & bit) ? -1 : 1;
    }
    return bit;
}
template<class Emit> void pile_transitions_wide(U128 c, Emit emit) {
    // Most states still fit the faster 64-bit transition arithmetic.
    if (!(c >> 60)) {
        pile_transitions(U64(c), [&](U64 key) { emit(U128(key)); });
        return;
    }
    U128 mask = ~U128(0) / 3, a = c & mask, z = (c >> 1) & mask;
    U128 ae = (a & 1) ^ 1, ze = (z & 1) ^ 1;
    emit((c << 2) | 3);
    if (a > 1) emit((ae << 1) | (a >> 2) | (z << 3));
    if (z > 1) emit(ze | (a << 2) | (z >> 1));
    if (a > 1 && z > 1 && (ae || ze)) {
        if (ae && !ze) a ^= walk_pile_wide(a);
        else if (ze && !ae) z ^= walk_pile_wide(z);
        emit((((z >> 2) << 3) | a) >> 2);
    }
}

U128 reflect_gap(U128 c) {
    int terminals = width(c) / 2 - 1;
    U128 gap = c & ((U128(1) << (terminals + 1)) - 1);
    c >>= terminals + 1;
    U128 reversed = 0;
    for (int i = 0; i < terminals; ++i) reversed = (reversed << 1) | ((c >> i) & 1);
    U128 reflected = reversed ^ ((U128(1) << (terminals + 1)) - 1);
    return (reflected << (terminals + 1)) | ((U128(1) << terminals) / gap);
}
bool half_gap(U128 c) {
    int terminals = width(c) / 2 - 1;
    if (!(c & (U128(3) << (terminals / 2)))) return false;
    c >>= terminals + 1;
    int size = width(c) - 1;
    std::vector<int> partner(size), open;
    for (int i = 0; i < size; ++i) {
        if ((c >> i) & 1) {
            if (open.empty()) throw std::runtime_error("invalid Dyck path");
            int other = open.back(); open.pop_back();
            partner[i] = other; partner[other] = i;
        } else open.push_back(i);
    }
    if (!open.empty()) throw std::runtime_error("unclosed Dyck path");
    return joined(partner);
}
U128 walk_alfa(U128 c) {
    int balance = 0; U128 bit = 1;
    while (balance >= 0) {
        bit <<= 1;
        if (!bit) throw std::overflow_error("gap walk overflow");
        balance += (c & bit) ? -1 : 1;
    }
    return bit;
}
U128 walk_zulu(U128 c) {
    int balance = 0; U128 bit = U128(1) << (width(c) - 2);
    while (balance >= 0) {
        bit >>= 1;
        if (!bit) throw std::runtime_error("invalid gap walk");
        balance += (c & bit) ? 1 : -1;
    }
    return bit;
}
template<class Emit> void gap_transitions(U128 c, Emit emit) {
    int t = width(c) / 2 - 1;
    U128 gap = c & ((U128(1) << (t + 1)) - 1);
    c >>= t + 1;
    emit(((c << 1) | (U128(1) << (t + 2))) << (t + 3) | (gap << 1));
    U128 a = walk_alfa(c), z = walk_zulu(c);
    if (gap > 1) emit((((c ^ a) | (U128(1) << (t + 1))) >> 1) << (t + 1) | (gap >> 1));
    if (gap < (U128(1) << t)) emit((c ^ z ^ (U128(1) << t)) << (t + 2) | (gap << 1));
    if (z > 1 && gap > 1 && gap < (U128(1) << t))
        emit((((c ^ a ^ z) >> 1) ^ (U128(1) << (t - 1))) << (t - 1) | (gap >> 1));
}

// A gap is a single set bit. Store its exponent in six bits rather than
// storing the entire mostly-zero gap field; the Dyck word occupies the rest.
// Reflection and transitions still use the independent strip-gap rules above.
U64 pack_gap(U128 c) {
    int t = width(c) / 2 - 1;
    U128 gap = c & ((U128(1) << (t + 1)) - 1);
    U128 word = c >> (t + 1);
    if (!gap || (gap & (gap - 1)) || width(word) > 58)
        throw std::runtime_error("gap key cannot be packed");
    return (U64(word) << 6) | (width(gap) - 1);
}
U128 unpack_gap(U64 key) {
    U128 word = key >> 6;
    return (word << width(word)) | (U128(1) << (key & 63));
}

int tail_layers = 1;
template<class K, class TailK = K, class Reflect, class Half, class Transitions>
U64 count(int n, bool orbit, K initial, U128 maximum, Reflect reflect, Half half, Transitions transitions) {
    Counts<K> current; current.insert(initial, 1);
    U64 doubled = 0;
    if (n <= 1) return (half(initial) + half(reflect(initial))) / 2;
    for (int layer = 1; layer < n; ++layer) {
        bool final = layer >= n - tail_layers;
        if (final) {
            struct Memo { TailK key = 0; U64 value = 0; int remaining = -1; };
            auto finish = [&](auto &&self, TailK key, int remaining) -> U64 {
                if (!remaining) return half(key) + half(reflect(key));
                if (remaining == 1) {
                    U64 weight = 0;
                    transitions(key, [&](TailK next) {
                        if (next < maximum) weight += half(next) + half(reflect(next));
                    });
                    return weight;
                }
                // Exact, bounded, per-thread memoization. A collision only
                // evicts an old entry; key and remaining depth must both match.
                // Keep shallow leaf traffic out of this cache. Reflection
                // preserves the terminal weight, so orbit representatives
                // can also share cached completion weights.
                if (orbit) key = std::min(key, reflect(key));
                static thread_local std::vector<Memo> memo(size_t(1) << 20);
                size_t slot = (hash_key(key) ^ mix(U64(remaining) + 1)) & (memo.size() - 1);
                Memo &entry = memo[slot];
                if (entry.key == key && entry.remaining == remaining) return entry.value;
                U64 weight = 0;
                transitions(key, [&](TailK next) {
                        if (next < maximum) weight += self(self, next, remaining - 1);
                    });
                entry = {key, weight, remaining};
                return weight;
            };
            std::atomic<size_t> processed{0};
            size_t total = current.slots.size();
            size_t chunks = (total + 65535) / 65536;
            std::vector<U64> saved_weights(chunks, 0);
            std::vector<unsigned char> saved(chunks, 0);
            std::string partial_path;
            if (const char *path = std::getenv("AXIS_PARTIAL_FILE")) partial_path = path;
            else if (n == 32) partial_path = "/tmp/axis_n32_tail" + std::to_string(tail_layers) + ".partials";
            std::ofstream partial_file;
            if (!partial_path.empty()) {
                U64 fingerprint = 0;
                #pragma omp parallel for reduction(^:fingerprint)
                for (size_t i = 0; i < total; ++i) {
                    const auto &s = current.slots[i];
                    if (s.key) fingerprint ^= mix(hash_key(s.key) ^ mix(s.count) ^ mix(i));
                }
                std::ifstream previous(partial_path);
                std::string magic;
                if (previous >> magic) {
                    int saved_n, saved_tail, saved_key_bytes;
                    size_t saved_size, saved_total; U64 saved_fingerprint;
                    if (!(previous >> saved_n >> saved_tail >> saved_key_bytes >> saved_size >> saved_total >> saved_fingerprint)
                        || magic != "AXIS1" || saved_n != n || saved_tail != tail_layers
                        || saved_key_bytes != sizeof(K) || saved_size != current.size
                        || saved_total != total || saved_fingerprint != fingerprint)
                        throw std::runtime_error("partial checkpoint belongs to a different prefix state");
                    size_t chunk; U64 weight;
                    while (previous >> chunk >> weight) {
                        if (chunk >= chunks || (saved[chunk] && saved_weights[chunk] != weight))
                            throw std::runtime_error("invalid partial checkpoint");
                        saved[chunk] = 1; saved_weights[chunk] = weight;
                    }
                }
                previous.close();
                // Rewrite complete records so an interrupted trailing write
                // cannot be concatenated with a later record on resume.
                partial_file.open(partial_path, std::ios::trunc);
                if (!partial_file) throw std::runtime_error("cannot write partial checkpoint");
                partial_file << "AXIS1 " << n << ' ' << tail_layers << ' ' << sizeof(K) << ' '
                             << current.size << ' ' << total << ' ' << fingerprint << '\n';
                for (size_t chunk = 0; chunk < chunks; ++chunk)
                    if (saved[chunk]) partial_file << chunk << ' ' << saved_weights[chunk] << '\n';
                partial_file.flush();
                std::cerr << "partial checkpoint: " << partial_path << '\n';
            }
            #pragma omp parallel for schedule(dynamic, 1) reduction(+:doubled)
            for (size_t chunk = 0; chunk < total; chunk += 65536) {
                size_t chunk_index = chunk / 65536;
                U64 partial = saved_weights[chunk_index];
                if (!saved[chunk_index]) {
                    for (size_t i = chunk; i < std::min(chunk + 65536, total); ++i) {
                        const auto &s = current.slots[i];
                        if (s.key) partial += s.count * finish(finish, s.key, n - layer);
                    }
                    if (partial_file.is_open()) {
                        #pragma omp critical(checkpoint_write)
                        {
                            partial_file << chunk_index << ' ' << partial << '\n';
                            partial_file.flush();
                        }
                    }
                }
                doubled += partial;
                size_t completed = processed.fetch_add(65536) + 65536;
                if (completed % (size_t(1) << 24) == 0) {
                    #pragma omp critical
                    std::cerr << "n=" << n << " finish=" << completed << '/' << total << '\n';
                }
            }
            break;
        }
        // Reserve before inserting: growing a large target would temporarily
        // retain its old allocation alongside both source and replacement.
        size_t capacity = 16;
        if (!final) while (capacity * 7 < current.size * 23) capacity *= 2;
        Counts<K> next(capacity);
        for (const auto &s: current.slots) if (s.key) {
            transitions(s.key, [&](TailK key) {
                if (key >= maximum) return;
                if (final) {
                    int weight = half(key) + half(reflect(key));
                    for (int j = 0; j < weight; ++j) doubled = add_checked(doubled, s.count);
                } else {
                    if (orbit) key = std::min(key, reflect(key));
                    if (key > std::numeric_limits<K>::max())
                        throw std::overflow_error("prefix key needs wider storage; increase tail-layers");
                    next.insert(K(key), s.count);
                }
            });
        }
        std::cerr << "n=" << n << " layer=" << layer << " states=" << next.size << '\n';
        current = std::move(next);
    }
    if (doubled & 1) throw std::runtime_error("nonintegral symmetry count");
    return doubled / 2;
}
int main(int argc, char **argv) {
    try {
        if (argc < 3 || argc > 4) throw std::runtime_error("usage: axis_transfer flow n [tail-layers]");
        std::string flow = argv[1]; int n = std::stoi(argv[2]);
        if (argc == 4) tail_layers = std::stoi(argv[3]);
        if (tail_layers < 1 || tail_layers > 10) throw std::runtime_error("tail-layers must be 1..10");
        // Pile keys need at most 2*n+2 bits. At n<=32 even the
        // doubled terminal sum is bounded by 2*4^31 and fits uint64_t.
        if (n < 0 || n > 32) throw std::runtime_error("supported indices: 0..32");
        if ((flow == "pile" || flow == "pile-plain") && n > 31)
            throw std::runtime_error("use pile-wide above n=31");
        if (flow != "pile" && flow != "pile-plain" && flow != "pile-wide" && n > 30)
            throw std::runtime_error("gap keys support n<=30");
        rlimit memory{U64(28) << 30, U64(28) << 30};
        if (setrlimit(RLIMIT_AS, &memory)) throw std::runtime_error("cannot set memory limit");
        auto start = std::chrono::steady_clock::now();
        bool orbit = flow.find("plain") == std::string::npos;
        U64 result;
        if (flow == "pile-wide") {
            result = count<U64, U128>(n, true, 15, U128(1) << (2 * n + 4),
                reflect_pile_wide, half_pile_wide,
                [](U128 c, auto emit) { pile_transitions_wide(c, emit); });
        } else if (flow == "pile" || flow == "pile-plain") {
            result = count<U64>(n, orbit, 15, U128(1) << (2 * n + 4), reflect_pile, half_pile,
                [](U64 c, auto emit) { pile_transitions(c, emit); });
        } else if (flow == "gap-compact") {
            if (n > 28) throw std::runtime_error("compact gap supports n<=28");
            U128 maximum = U128(1) << (4 * n + 4);
            result = count<U64>(n, true, pack_gap(50), std::numeric_limits<U64>::max(),
                [](U64 c) { return pack_gap(reflect_gap(unpack_gap(c))); },
                [](U64 c) { return half_gap(unpack_gap(c)); },
                [maximum](U64 c, auto emit) {
                    gap_transitions(unpack_gap(c), [&](U128 key) {
                        if (key < maximum) emit(pack_gap(key));
                    });
                });
        } else if (flow == "gap" || flow == "gap-plain") {
            result = count<U128>(n, orbit, 50, U128(1) << (4 * n + 4), reflect_gap, half_gap,
                [](U128 c, auto emit) { gap_transitions(c, emit); });
        } else throw std::runtime_error("unknown flow");
        rusage usage{}; getrusage(RUSAGE_SELF, &usage);
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "{\"flow\":\"cpp-" << flow << "\",\"n\":" << n << ",\"value\":" << result
                  << ",\"tail_layers\":" << tail_layers << ",\"seconds\":" << seconds << ",\"max_rss_kib\":" << usage.ru_maxrss << "}\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
