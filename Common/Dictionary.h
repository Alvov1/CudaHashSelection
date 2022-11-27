#ifndef HASHSELECTION_DICTIONARY_H
#define HASHSELECTION_DICTIONARY_H

#include <random>

#include "Word.h"
#include "ReplacementDictionary.h"

template <typename Char = char>
class Dictionary final {
    static const Word<Char> words[];
    Dictionary() = default;
public:
    static size_t size();
    static Word<Char> get(unsigned index);
    static Word<Char> getRandom();
    static void find(const std::basic_string<Char>& hash);

    Dictionary(const Dictionary& copy) = delete;
    Dictionary& operator=(const Dictionary& assign) = delete;
    Dictionary(Dictionary&& move) = delete;
    Dictionary& operator=(Dictionary&& moveAssign) = delete;
};

template <typename Char>
Word<Char> Dictionary<Char>::getRandom() {
//    static std::random_device dev;
//    static std::mt19937 rng(dev());
//    static std::uniform_int_distribution<std::mt19937::result_type> dist6(0, size() - 1);

    const auto index = 1;//dist6(rng);
    return words[index];
}

template <typename Char>
Word<Char> Dictionary<Char>::get(unsigned int index) {
    if(index > size())
        throw std::invalid_argument("Index is out of range.");
    return words[index];
}

template <>
constexpr const Word<char> Dictionary<char>::words[] = {
        { "a" }, { "about" }, { "act" }, { "actually" }, { "add" }, { "after" }, { "again" },
        { "against" }, { "age" }, { "ago" }, { "air" }, { "all" }, { "also" }, { "always" },
        { "am" }, { "among" }, { "an" }, { "and" }, { "animal" }, { "another" }, { "answer" },
        { "appear" }, { "are" }, { "area" }, { "as" }, { "ask" }, { "at" }, { "back" },
        { "ball" }, { "base" }, { "be" }, { "beauty" }, { "because" }, { "become" }, { "bed" },
        { "been" }, { "before" }, { "began" }, { "begin" }, { "behind" }, { "best" }, { "better" },
        { "better" }, { "between" }, { "big" }, { "bird" }, { "black" }, { "blue" }, { "boat" },
        { "body" }, { "book" }, { "both" }, { "bottom" }, { "box" }, { "boy" }, { "bring" },
        { "brought" }, { "build" }, { "built" }, { "busy" }, { "but" }, { "by" }, { "call" },
        { "came" }, { "can" }, { "car" }, { "care" }, { "carefully" }, { "carry" }, { "centre" },
        { "certain" }, { "change" }, { "check" }, { "child" }, { "children" }, { "city" }, { "class" },
        { "clear" }, { "close" }, { "cold" }, { "colour" }, { "come" }, { "common" }, { "community" },
        { "complete" }, { "contain" }, { "could" }, { "country" }, { "course" }, { "create" },
        { "cried" },{ "cross" }, { "cry" }, { "cut" }, { "dark" }, { "day" }, { "decide" },
        { "decided" }, { "deep" },{ "develop" }, { "did" }, { "didn’t" }, { "different" },
        { "do" }, { "does" }, { "dog" }, { "don’t" }, { "door" }, { "down" }, { "draw" },
        { "dream" }, { "drive" }, { "dry" }, { "during" }, { "each" }, { "early" }, { "earth" },
        { "east" }, { "easy" }, { "eat" }, { "effort" }, { "enough" }, { "every" }, { "example" },
        { "experience" }, { "explain" }, { "eye" }, { "face" }, { "fact" }, { "false" }, { "family" },
        { "far" }, { "farm" }, { "fast" }, { "father" }, { "feel" }, { "feet" }, { "few" },
        { "field" },{ "find" }, { "fire" }, { "first" }, { "fish" }, { "five" }, { "fly" },
        { "follow" }, { "food" },{ "form" }, { "found" }, { "four" }, { "friend" }, { "from" },
        { "front" }, { "full" }, { "game" }, { "gave" }, { "get" }, { "girl" }, { "give" },
        { "go" }, { "gold" }, { "good" }, { "got" }, { "government" }, { "great" }, { "green" },
        { "ground" }, { "group" }, { "grow" }, { "guy" }, { "had" }, { "half" }, { "hand" },
        { "happen" }, { "happened" }, { "hard" }, { "has" }, { "have" }, { "he" }, { "hear" },
        { "heat" }, { "heavy" }, { "help" }, { "her" }, { "here" }, { "high" }, { "his" },
        { "hold" }, { "home" }, { "horse" }, { "hot" }, { "hour" }, { "house" }, { "hundred" },
        { "idea" }, { "if" }, { "important" }, { "in" }, { "inch" }, { "include" }, { "into" },
        { "is" }, { "island" }, { "it" }, { "just" }, { "keep" }, { "kind" }, { "king" },
        { "knew" }, { "know" }, { "known" }, { "land" }, { "language" }, { "large" }, { "last" },
        { "late" }, { "later" }, { "laugh" }, { "lead" }, { "learn" }, { "leave" }, { "left" },
        { "less" }, { "less" }, { "let" }, { "letter" }, { "life" }, { "light" }, { "like" },
        { "line" }, { "list" }, { "listen" }, { "little" }, { "live" }, { "long" }, { "look" },
        { "love" }, { "low" }, { "machine" }, { "made" }, { "make" }, { "man" }, { "many" },
        { "map" }, { "mark" }, { "may" }, { "mean" }, { "measure" }, { "men" }, { "might" },
        { "mile" }, { "million" }, { "mind" }, { "minute" }, { "miss" }, { "money" }, { "month" },
        { "moon" }, { "more" }, { "more" }, { "morning" }, { "most" }, { "mother" }, { "mountain" },
        { "move" }, { "much" }, { "music" }, { "must" }, { "my" }, { "name" }, { "nation" },
        { "near" }, { "need" }, { "never" }, { "new" }, { "next" }, { "night" }, { "no" },
        { "north" }, { "note" }, { "notice" }, { "noun" }, { "now" }, { "number" }, { "object" },
        { "of" }, { "off" }, { "office" }, { "often" }, { "oh" }, { "oil" }, { "old" },
        { "on" }, { "once" }, { "one" }, { "only" }, { "open" }, { "or" }, { "order" },
        { "other" }, { "our" }, { "out" }, { "over" }, { "page" }, { "pair" }, { "part" },
        { "pass" }, { "passed" }, { "people" }, { "perhaps" }, { "person" }, { "picture" }, { "place" },
        { "plan" }, { "plane" }, { "plant" }, { "play" }, { "point" }, { "power" }, { "probably" },
        { "problem" }, { "product" }, { "provide" }, { "pull" }, { "put" }, { "question" }, { "quick" },
        { "rain" }, { "ran" }, { "reach" }, { "read" }, { "ready" }, { "real" }, { "receive" },
        { "record" }, { "red" }, { "relationship" }, { "remember" }, { "right" }, { "river" },
        { "road" }, { "rock" }, { "room" }, { "round" }, { "rule" }, { "run" }, { "said" },
        { "same" }, { "saw" }, { "say" }, { "school" }, { "science" }, { "sea" }, { "season" },
        { "second" }, { "see" }, { "seem" }, { "self" }, { "sentence" }, { "serve" }, { "set" },
        { "several" }, { "shape" }, { "she" }, { "ship" }, { "short" }, { "should" }, { "show" },
        { "shown" }, { "side" }, { "simple" }, { "since" }, { "sing" }, { "sit" }, { "six" },
        { "size" }, { "sleep" }, { "slow" }, { "small" }, { "snow" }, { "so" }, { "some" },
        { "something" }, { "song" }, { "soon" }, { "sound" }, { "south" }, { "space" }, { "special" },
        { "spell" }, { "spring" }, { "stand" }, { "star" }, { "start" }, { "stay" }, { "step" },
        { "stood" }, { "stop" }, { "story" }, { "street" }, { "strong" }, { "study" }, { "such" },
        { "summer" }, { "sun" }, { "system" }, { "table" }, { "take" }, { "talk" }, { "teach" },
        { "tell" }, { "ten" }, { "test" }, { "than" }, { "that" }, { "the" }, { "their" },
        { "them" }, { "then" }, { "there" }, { "these" }, { "they" }, { "thing" }, { "think" },
        { "this" }, { "those" }, { "though" }, { "thought" }, { "thousand" }, { "three" }, { "through" },
        { "time" }, { "to" }, { "together" }, { "told" }, { "too" }, { "took" }, { "top" },
        { "toward" }, { "town" }, { "travel" }, { "tree" }, { "try" }, { "true" }, { "turn" },
        { "two" }, { "under" }, { "understand" }, { "until" }, { "up" }, { "upon" }, { "us" },
        { "use" }, { "usual" }, { "very" }, { "voice" }, { "vowel" }, { "wait" }, { "walk" },
        { "want" }, { "war" }, { "warm" }, { "was" }, { "watch" }, { "water" }, { "wave" },
        { "way" }, { "we" }, { "week" }, { "weight" }, { "were" }, { "west" }, { "what" },
        { "wheel" }, { "where" }, { "which" }, { "white" }, { "who" }, { "why" }, { "will" },
        { "wind" }, { "winter" }, { "with" }, { "without" }, { "woman" }, { "wonder" },
        { "wood" }, { "word" }, { "words" }, { "work" }, { "world" }, { "would" }, { "write" },
        { "wrong" }, { "year" }, { "yes" }, { "you" }, { "young" }
};

template <>
constexpr const Word<wchar_t> Dictionary<wchar_t>::words[] = {
        { L"a" }, { L"about" }, { L"act" }, { L"actually" }, { L"add" }, { L"after" }, { L"again" },
        { L"against" }, { L"age" }, { L"ago" }, { L"air" }, { L"all" }, { L"also" }, { L"always" },
        { L"am" }, { L"among" }, { L"an" }, { L"and" }, { L"animal" }, { L"another" }, { L"answer" },
        { L"appear" }, { L"are" }, { L"area" }, { L"as" }, { L"ask" }, { L"at" }, { L"back" },
        { L"ball" }, { L"base" }, { L"be" }, { L"beauty" }, { L"because" }, { L"become" }, { L"bed" },
        { L"been" }, { L"before" }, { L"began" }, { L"begin" }, { L"behind" }, { L"best" }, { L"better" },
        { L"better" }, { L"between" }, { L"big" }, { L"bird" }, { L"black" }, { L"blue" }, { L"boat" },
        { L"body" }, { L"book" }, { L"both" }, { L"bottom" }, { L"box" }, { L"boy" }, { L"bring" },
        { L"brought" }, { L"build" }, { L"built" }, { L"busy" }, { L"but" }, { L"by" }, { L"call" },
        { L"came" }, { L"can" }, { L"car" }, { L"care" }, { L"carefully" }, { L"carry" }, { L"centre" },
        { L"certain" }, { L"change" }, { L"check" }, { L"child" }, { L"children" }, { L"city" }, { L"class" },
        { L"clear" }, { L"close" }, { L"cold" }, { L"colour" }, { L"come" }, { L"common" }, { L"community" },
        { L"complete" }, { L"contain" }, { L"could" }, { L"country" }, { L"course" }, { L"create" },
        { L"cried" },{ L"cross" }, { L"cry" }, { L"cut" }, { L"dark" }, { L"day" }, { L"decide" },
        { L"decided" }, { L"deep" },{ L"develop" }, { L"did" }, { L"didn’t" }, { L"different" },
        { L"do" }, { L"does" }, { L"dog" }, { L"don’t" }, { L"door" }, { L"down" }, { L"draw" },
        { L"dream" }, { L"drive" }, { L"dry" }, { L"during" }, { L"each" }, { L"early" }, { L"earth" },
        { L"east" }, { L"easy" }, { L"eat" }, { L"effort" }, { L"enough" }, { L"every" }, { L"example" },
        { L"experience" }, { L"explain" }, { L"eye" }, { L"face" }, { L"fact" }, { L"false" }, { L"family" },
        { L"far" }, { L"farm" }, { L"fast" }, { L"father" }, { L"feel" }, { L"feet" }, { L"few" },
        { L"field" },{ L"find" }, { L"fire" }, { L"first" }, { L"fish" }, { L"five" }, { L"fly" },
        { L"follow" }, { L"food" },{ L"form" }, { L"found" }, { L"four" }, { L"friend" }, { L"from" },
        { L"front" }, { L"full" }, { L"game" }, { L"gave" }, { L"get" }, { L"girl" }, { L"give" },
        { L"go" }, { L"gold" }, { L"good" }, { L"got" }, { L"government" }, { L"great" }, { L"green" },
        { L"ground" }, { L"group" }, { L"grow" }, { L"guy" }, { L"had" }, { L"half" }, { L"hand" },
        { L"happen" }, { L"happened" }, { L"hard" }, { L"has" }, { L"have" }, { L"he" }, { L"hear" },
        { L"heat" }, { L"heavy" }, { L"help" }, { L"her" }, { L"here" }, { L"high" }, { L"his" },
        { L"hold" }, { L"home" }, { L"horse" }, { L"hot" }, { L"hour" }, { L"house" }, { L"hundred" },
        { L"idea" }, { L"if" }, { L"important" }, { L"in" }, { L"inch" }, { L"include" }, { L"into" },
        { L"is" }, { L"island" }, { L"it" }, { L"just" }, { L"keep" }, { L"kind" }, { L"king" },
        { L"knew" }, { L"know" }, { L"known" }, { L"land" }, { L"language" }, { L"large" }, { L"last" },
        { L"late" }, { L"later" }, { L"laugh" }, { L"lead" }, { L"learn" }, { L"leave" }, { L"left" },
        { L"less" }, { L"less" }, { L"let" }, { L"letter" }, { L"life" }, { L"light" }, { L"like" },
        { L"line" }, { L"list" }, { L"listen" }, { L"little" }, { L"live" }, { L"long" }, { L"look" },
        { L"love" }, { L"low" }, { L"machine" }, { L"made" }, { L"make" }, { L"man" }, { L"many" },
        { L"map" }, { L"mark" }, { L"may" }, { L"mean" }, { L"measure" }, { L"men" }, { L"might" },
        { L"mile" }, { L"million" }, { L"mind" }, { L"minute" }, { L"miss" }, { L"money" }, { L"month" },
        { L"moon" }, { L"more" }, { L"more" }, { L"morning" }, { L"most" }, { L"mother" }, { L"mountain" },
        { L"move" }, { L"much" }, { L"music" }, { L"must" }, { L"my" }, { L"name" }, { L"nation" },
        { L"near" }, { L"need" }, { L"never" }, { L"new" }, { L"next" }, { L"night" }, { L"no" },
        { L"north" }, { L"note" }, { L"notice" }, { L"noun" }, { L"now" }, { L"number" }, { L"object" },
        { L"of" }, { L"off" }, { L"office" }, { L"often" }, { L"oh" }, { L"oil" }, { L"old" },
        { L"on" }, { L"once" }, { L"one" }, { L"only" }, { L"open" }, { L"or" }, { L"order" },
        { L"other" }, { L"our" }, { L"out" }, { L"over" }, { L"page" }, { L"pair" }, { L"part" },
        { L"pass" }, { L"passed" }, { L"people" }, { L"perhaps" }, { L"person" }, { L"picture" }, { L"place" },
        { L"plan" }, { L"plane" }, { L"plant" }, { L"play" }, { L"point" }, { L"power" }, { L"probably" },
        { L"problem" }, { L"product" }, { L"provide" }, { L"pull" }, { L"put" }, { L"question" }, { L"quick" },
        { L"rain" }, { L"ran" }, { L"reach" }, { L"read" }, { L"ready" }, { L"real" }, { L"receive" },
        { L"record" }, { L"red" }, { L"relationship" }, { L"remember" }, { L"right" }, { L"river" },
        { L"road" }, { L"rock" }, { L"room" }, { L"round" }, { L"rule" }, { L"run" }, { L"said" },
        { L"same" }, { L"saw" }, { L"say" }, { L"school" }, { L"science" }, { L"sea" }, { L"season" },
        { L"second" }, { L"see" }, { L"seem" }, { L"self" }, { L"sentence" }, { L"serve" }, { L"set" },
        { L"several" }, { L"shape" }, { L"she" }, { L"ship" }, { L"short" }, { L"should" }, { L"show" },
        { L"shown" }, { L"side" }, { L"simple" }, { L"since" }, { L"sing" }, { L"sit" }, { L"six" },
        { L"size" }, { L"sleep" }, { L"slow" }, { L"small" }, { L"snow" }, { L"so" }, { L"some" },
        { L"something" }, { L"song" }, { L"soon" }, { L"sound" }, { L"south" }, { L"space" }, { L"special" },
        { L"spell" }, { L"spring" }, { L"stand" }, { L"star" }, { L"start" }, { L"stay" }, { L"step" },
        { L"stood" }, { L"stop" }, { L"story" }, { L"street" }, { L"strong" }, { L"study" }, { L"such" },
        { L"summer" }, { L"sun" }, { L"system" }, { L"table" }, { L"take" }, { L"talk" }, { L"teach" },
        { L"tell" }, { L"ten" }, { L"test" }, { L"than" }, { L"that" }, { L"the" }, { L"their" },
        { L"them" }, { L"then" }, { L"there" }, { L"these" }, { L"they" }, { L"thing" }, { L"think" },
        { L"this" }, { L"those" }, { L"though" }, { L"thought" }, { L"thousand" }, { L"three" }, { L"through" },
        { L"time" }, { L"to" }, { L"together" }, { L"told" }, { L"too" }, { L"took" }, { L"top" },
        { L"toward" }, { L"town" }, { L"travel" }, { L"tree" }, { L"try" }, { L"true" }, { L"turn" },
        { L"two" }, { L"under" }, { L"understand" }, { L"until" }, { L"up" }, { L"upon" }, { L"us" },
        { L"use" }, { L"usual" }, { L"very" }, { L"voice" }, { L"vowel" }, { L"wait" }, { L"walk" },
        { L"want" }, { L"war" }, { L"warm" }, { L"was" }, { L"watch" }, { L"water" }, { L"wave" },
        { L"way" }, { L"we" }, { L"week" }, { L"weight" }, { L"were" }, { L"west" }, { L"what" },
        { L"wheel" }, { L"where" }, { L"which" }, { L"white" }, { L"who" }, { L"why" }, { L"will" },
        { L"wind" }, { L"winter" }, { L"with" }, { L"without" }, { L"woman" }, { L"wonder" },
        { L"wood" }, { L"word" }, { L"words" }, { L"work" }, { L"world" }, { L"would" }, { L"write" },
        { L"wrong" }, { L"year" }, { L"yes" }, { L"you" }, { L"young" }
};

template <typename Char>
size_t Dictionary<Char>::size() {
    return sizeof(words) / sizeof(Word<Char>);
}

template<typename Char>
void Dictionary<Char>::find(const std::basic_string<Char>& hash) {
    static typename ReplacementDictionary<Char>::Comparator closure =
            [](const std::basic_string<Char>& current, const std::basic_string<Char>& requiredHash) {
                HostSHA256 currentHash(current.c_str(), current.size());
                return currentHash.to_string() == requiredHash;
            };

    for(unsigned i = 0; i < Dictionary<Char>::size(); ++i) {
        const auto current = Dictionary<Char>::get(i);

        const auto result = ReplacementDictionary<Char>::enumerate(current, hash, closure);
        if(result.has_value())
            Timer::out << "Found a coincidence with word " << result.value() << L"." << Timer::endl;
    }
}

#endif //HASHSELECTION_DICTIONARY_H
