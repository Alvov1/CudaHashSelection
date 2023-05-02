#include "DeviceHash.h"

namespace Substitutions {
    u32 rotateRight(u32 value, u32 bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    u32 choose(u32 first, u32 second, u32 third) {
        return (first & second) ^ (~first & third);
    }

    u32 majority(u32 first, u32 second, u32 third) {
        return (first & (second | third)) | (second & third);
    }

    u32 sig0(u32 value) {
        return rotateRight(value, 7) ^ rotateRight(value, 18) ^ (value >> 3);
    }

    u32 sig1(u32 value) {
        return rotateRight(value, 17) ^ rotateRight(value, 19) ^ (value >> 10);
    }
}

template <typename Char>
DeviceSHA256::DeviceSHA256(const Char* input, std::size_t length) {
    std::array<u32, 8> state = {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    u32 blockLen {};
    u64 bitLen {};

    for(u32 i = 0; i < length; ++i) {
        data[blockLen++] = input[i];
        if(blockLen == 64) {
            transform(state);
            bitLen += 512;
            blockLen = 0;
        }
    }

    u64 position = blockLen;
    u8 end = blockLen < 56 ? 56 : 64;

    data[position++] = 0x80;
    for(; position < end; ++position)
        data[position] = 0x00;

    if(blockLen > 55) {
        transform(state);
        std::fill(data.begin(), data.begin() + 56, 0);
    }

    bitLen += blockLen * 8;
    data[63] = bitLen;
    data[62] = bitLen >> 8;
    data[61] = bitLen >> 16;
    data[60] = bitLen >> 24;
    data[59] = bitLen >> 32;
    data[58] = bitLen >> 40;
    data[57] = bitLen >> 48;
    data[56] = bitLen >> 56;
    transform(state);

    for(u8 i = 0; i < 4; ++i)
        for(u8 j = 0; j < 8; ++j)
            data[i + j * 4] = (state[j] >> (24 - i * 8)) & 0x0000'00ff;
}
template DeviceSHA256::DeviceSHA256(const char* data, std::size_t length);
template DeviceSHA256::DeviceSHA256(const wchar_t* data, std::size_t length);

void DeviceSHA256::transform(std::array<u32, 8>& state) {
    std::array<u32, 64> m {};
    std::array<u32, 8> tState {};

    for(u8 i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);

    for(u8 i = 16; i < 64; ++i)
        m[i] = Substitutions::sig1(m[i - 2]) + m[i - 7] + Substitutions::sig0(m[i - 15]) + m[i - 16];

    for(u8 i = 0; i < 8; ++i)
        tState[i] = state[i];

    for(u8 i = 0; i < 64; ++i) {
        static constexpr std::array<u32, 64> messageSchedule = {
                0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };

        u32 addition  = m[i] + messageSchedule[i] + tState[7] + Substitutions::choose(tState[4], tState[5], tState[6]) +
                        (Substitutions::rotateRight(tState[4], 6) ^ Substitutions::rotateRight(tState[4], 11) ^
                    Substitutions::rotateRight(tState[4], 25));
        u32 start = (Substitutions::rotateRight(tState[0], 2) ^ Substitutions::rotateRight(tState[0], 13) ^
                     Substitutions::rotateRight(tState[0], 22))
                    + Substitutions::majority(tState[0], tState[1], tState[2]) + addition;

        tState[7] = tState[6];
        tState[6] = tState[5];
        tState[5] = tState[4];
        tState[4] = tState[3] + addition;
        tState[3] = tState[2];
        tState[2] = tState[1];
        tState[1] = tState[0];
        tState[0] = start;
    }

    for(u8 i = 0; i < 8; ++i)
        state[i] += tState[i];
}

std::string DeviceSHA256::to_string() const {
    std::stringstream s;
    s << std::setfill('0') << std::hex;

    for(uint8_t i = 0 ; i < 32 ; i++)
        s << std::setw(2) << static_cast<unsigned short>(data[i]);
    return s.str();
}
