#include "DeviceHash.h"

DEVICE void DeviceSHA256::finalDigest() {
    unsigned block_nb = (1 + ((DeviceSHA256::BlockSize - 9) < (len_ % DeviceSHA256::BlockSize)));
    unsigned len_b = (tot_len_ + len_) << 3;
    unsigned int pm_len = block_nb << 6;
    memset(block_ + len_, 0, pm_len - len_);
    block_[len_] = 0x80;
    SHA2_UNPACK32(len_b, block_ + pm_len - 4);
    transform_(block_, block_nb);

    for(auto i = 0; i < 8; i++) {
        auto number = digest_[i];
        auto position = i * 8 + 7;
        for(auto j = 0; j < 8; ++j, number /= 16) {
            auto curValue = number % 16;
            if(curValue < 10)
                result[position--] = static_cast<char>('0' + curValue);
            else
                result[position--] = static_cast<char>('a' + curValue - 10);
        }
    }
}

DEVICE void DeviceSHA256::update_(unsigned char const *message, size_t len) {
    unsigned int block_nb;
    unsigned int new_len, rem_len, tmp_len;
    const unsigned char *shifted_message;
    tmp_len = DeviceSHA256::BlockSize - len_;
    rem_len = len < tmp_len ? len : tmp_len;
    memcpy(&block_[len_], message, rem_len);
    if (len_ + len < DeviceSHA256::BlockSize) {
        len_ += len;
        return;
    }
    new_len = len - rem_len;
    block_nb = new_len / DeviceSHA256::BlockSize;
    shifted_message = message + rem_len;
    transform_(block_, 1);
    transform_(shifted_message, block_nb);
    rem_len = new_len % DeviceSHA256::BlockSize;
    memcpy(block_, &shifted_message[block_nb << 6], rem_len);
    len_ = rem_len;
    tot_len_ += (block_nb + 1) << 6;
}

DEVICE void DeviceSHA256::transform_(unsigned char const *message, unsigned int block_nb) {
    uint32_t w[64];
    uint32_t wv[8];
    uint32_t t1, t2;
    const unsigned char *sub_block;
    int i;
    int j;
    for (i = 0; i < (int) block_nb; i++) {
        sub_block = message + (i << 6);
        for (j = 0; j < 16; j++) {
            w[j] = SHA2_PACK32(&sub_block[j << 2]);
        }
        for (j = 16; j < 64; j++) {
            w[j] = SHA256_F4(w[j - 2]) + w[j - 7] + SHA256_F3(w[j - 15]) + w[j - 16];
        }
        for (j = 0; j < 8; j++) {
            wv[j] = digest_[j];
        }
        for (j = 0; j < 64; j++) {
            t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
                 + sha256_k(j) + w[j];
            t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
            wv[7] = wv[6];
            wv[6] = wv[5];
            wv[5] = wv[4];
            wv[4] = wv[3] + t1;
            wv[3] = wv[2];
            wv[2] = wv[1];
            wv[1] = wv[0];
            wv[0] = t1 + t2;
        }
        for (j = 0; j < 8; j++) {
            digest_[j] += wv[j];
        }
    }
}

DEVICE constexpr unsigned DeviceSHA256::sha256_k(size_t index) {
    unsigned constValues[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };
    return constValues[index];
}