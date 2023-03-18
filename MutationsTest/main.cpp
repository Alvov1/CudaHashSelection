#include <iostream>
#include <array>

const std::array<std::string_view, 26>& get() {
    static constexpr std::array<std::string_view, 26> words = {
            /* A */ "4@^",  /* B */ "86",     /* C */ "[<(",  /* D */ "",       /* E */ "3&",
            /* F */ "v",    /* G */ "6&9",    /* H */ "#",    /* I */ "1|/\\!", /* J */ "]}",
            /* K */ "(<x",  /* L */ "!127|",  /* M */ "",     /* N */ "^",      /* O */ "0",
            /* P */ "9?",   /* Q */ "20&9",   /* R */ "972",  /* S */ "3$z2",   /* T */ "7+",
            /* U */ "v",    /* V */ "u",      /* W */ "v",    /* X */ "%",      /* Y */ "j", /* Z */ "27s"
    };

    return words;
}

template <std::size_t size = 20>
void process(const std::string_view& view, const std::array<char, size>& buffer) {

}

int main() {

}