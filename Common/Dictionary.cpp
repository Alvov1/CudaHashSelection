#include "Dictionary.h"

constexpr Word Dictionary::dictionary[] = {
        { L"be" }, { L"and" }, { L"of" }, { L"a" }, { L"in" }, { L"to" }, { L"have" }, { L"too" }, { L"it" }, { L"I" }, { L"that" }, { L"for" }, { L"you" }, { L"he" }, { L"with" }, { L"on" }, { L"do" },
        { L"say" }, { L"this" }, { L"they" }, { L"at" }, { L"but" }, { L"we" }, { L"his" }, { L"from" }, { L"that" }, { L"not" }, { L"can’t" }, { L"won’t" }, { L"by" }, { L"she" }, { L"or" },
        { L"as" }, { L"what" }, { L"go" }, { L"their" }, { L"can" }, { L"who" }, { L"get" }, { L"if" }, { L"would" }, { L"her" }, { L"all" }, { L"my" }, { L"make" }, { L"about" }, { L"know" },
        { L"will" }, { L"as" }, { L"up" }, { L"one" }, { L"time" }, { L"there" }, { L"year" }, { L"so" }, { L"think" }, { L"when" }, { L"which" }, { L"them" }, { L"some" }, { L"me" },
        { L"people" }, { L"take" }, { L"out" }, { L"into" }, { L"just" }, { L"see" }, { L"him" }, { L"your" }, { L"come" }, { L"could" }, { L"now" }, { L"than" }, { L"like" }, { L"other" },
        { L"how" }, { L"then" }, { L"its" }, { L"our" }, { L"two" }, { L"more" }, { L"these" }, { L"want" }, { L"way" }, { L"look" }, { L"first" }, { L"also" }, { L"new" }, { L"because" },
        { L"day" }, { L"more" }, { L"use" }, { L"no" }, { L"man" }, { L"find" }, { L"here" }, { L"thing" }, { L"give" }, { L"many" }, { L"well" }, { L"only" }, { L"those" }, { L"tell" },
        { L"one" }, { L"very" }, { L"her" }, { L"even" }, { L"back" }, { L"any" }, { L"good" }, { L"woman" }, { L"through" }, { L"us" }, { L"life" }, { L"child" }, { L"there" }, { L"work" },
        { L"down" }, { L"may" }, { L"after" }, { L"should" }, { L"call" }, { L"world" }, { L"over" }, { L"school" }, { L"still" }, { L"try" }, { L"in" }, { L"as" }, { L"last" }, { L"ask" },
        { L"need" }, { L"too" }, { L"feel" }, { L"three" }, { L"when" }, { L"state" }, { L"never" }, { L"become" }, { L"between" }, { L"high" }, { L"really" }, { L"something" },
        { L"most" }, { L"another" }, { L"much" }, { L"family" }, { L"own" }, { L"out" }, { L"leave" }, { L"put" }, { L"old" }, { L"while" }, { L"mean" }, { L"on" }, { L"keep" },
        { L"student" }, { L"why" }, { L"let" }, { L"great" }, { L"same" }, { L"big" }, { L"group" }, { L"begin" }, { L"seem" }, { L"country" }, { L"help" }, { L"talk" }, { L"where" },
        { L"turn" }, { L"problem" }, { L"every" }, { L"start" }, { L"hand" }, { L"might" }, { L"American" }, { L"show" }, { L"part" }, { L"about" }, { L"against" }, { L"place" },
        { L"over" }, { L"such" }, { L"again" }, { L"few" }, { L"case" }, { L"most" }, { L"week" }, { L"company" }, { L"where" }, { L"system" }, { L"each" }, { L"right" },
        { L"program" }, { L"hear" }, { L"so" }, { L"question" }, { L"during" }, { L"work" }, { L"play" }, { L"government" }, { L"run" }, { L"small" }, { L"number" }, { L"off" },
        { L"always" }, { L"move" }, { L"like" }, { L"night" }, { L"live" }, { L"Mr." }, { L"point" }, { L"believe" }, { L"hold" }, { L"today" }, { L"bring" }, { L"happen" },
        { L"next" }, { L"without" }, { L"before" }, { L"large" }, { L"all" }, { L"million" }, { L"must" }, { L"home" }, { L"under" }, { L"water" }, { L"room" }, { L"write" },
        { L"mother" }, { L"area" }, { L"national" }, { L"money" }, { L"story" }, { L"young" }, { L"fact" }, { L"month" }, { L"different" }, { L"lot" }, { L"right" }, { L"study" },
        { L"book" }, { L"eye" }, { L"job" }, { L"ptr" }, { L"though" }, { L"business" }, { L"issue" }, { L"side" }, { L"kind" }, { L"four" }, { L"head" }, { L"far" }, { L"black" },
        { L"long" }, { L"both" }, { L"little" }, { L"house" }, { L"yes" }, { L"after" }, { L"since" }, { L"long" }, { L"provide" }, { L"service" }, { L"around" }, { L"friend" },
        { L"important" }, { L"father" }, { L"sit" }, { L"away" }, { L"until" }, { L"power" }, { L"hour" }, { L"game" }, { L"often" }, { L"yet" }, { L"line" }, { L"political" },
        { L"end" }, { L"among" }, { L"ever" }, { L"stand" }, { L"bad" }, { L"lose" }, { L"however" }, { L"member" }, { L"pay" }, { L"law" }, { L"meet" }, { L"car" }, { L"city" },
        { L"almost" }, { L"include" }, { L"continue" }, { L"set" }, { L"later" }, { L"community" }, { L"much" }, { L"name" }, { L"five" }, { L"once" }, { L"white" }, { L"least" },
        { L"president " }, { L"learn" }, { L"real" }, { L"change" }, { L"team" }, { L"minute" }, { L"best" }, { L"several" }, { L"idea" }, { L"kid" }, { L"body" },
        { L"information" }, { L"nothing" }, { L"ago" }, { L"right" }, { L"lead" }, { L"social" }, { L"understand" }, { L"whether" }, { L"back" }, { L"watch" },
        { L"together" }, { L"follow" }, { L"around" }, { L"parent" }, { L"only" }, { L"stop" }, { L"face" }, { L"anything" }, { L"create" }, { L"public" }, { L"already" },
        { L"speak" }, { L"others" }, { L"read" }, { L"level" }, { L"allow" }, { L"add" }, { L"office" }, { L"spend" }, { L"door" }, { L"health" }, { L"person" }, { L"art" },
        { L"sure" }, { L"such" }, { L"war" }, { L"history" }, { L"party" }, { L"within" }, { L"grow" }, { L"result" }, { L"open" }, { L"change" }, { L"morning" }, { L"walk" },
        { L"reason" }, { L"low" }, { L"win" }, { L"research" }, { L"girl" }, { L"guy" }, { L"early" }, { L"food" }, { L"before" }, { L"moment" }, { L"himself" }, { L"air" },
        { L"teacher" }, { L"force" }, { L"offer" }, { L"enough" }, { L"both" }, { L"education" }, { L"across" }, { L"although" }, { L"remember" }, { L"foot" }, { L"second" },
        { L"boy" }, { L"maybe" }, { L"toward" }, { L"able" }, { L"age" }, { L"off" }, { L"policy" }, { L"everything" }, { L"love" }, { L"process" }, { L"music" }, { L"including" },
        { L"consider" }, { L"appear" }, { L"actually" }, { L"buy" }, { L"probably" }, { L"human" }, { L"wait" }, { L"serve" }, { L"market" }, { L"die" }, { L"send" },
        { L"expect" }, { L"home" }, { L"sense" }, { L"build" }, { L"stay" }, { L"fall" }, { L"oh" }, { L"nation" }, { L"plan" }, { L"cut" }, { L"college" }, { L"interest" },
        { L"death" }, { L"course" }, { L"someone" }, { L"experience" }, { L"behind" }, { L"reach" }, { L"local" }, { L"kill" }, { L"six" }, { L"remain" }, { L"effect" },
        { L"use" }, { L"yeah" }, { L"suggest" }, { L"class" }, { L"control" }, { L"raise" }, { L"care" }, { L"perhaps" }, { L"little" }, { L"late" }, { L"hard" }, { L"field" },
        { L"else" }, { L"pass" }, { L"former" }, { L"sell" }, { L"major" }, { L"sometimes" }, { L"require" }, { L"along" }, { L"development" }, { L"themselves" },
        { L"report" }, { L"role" }, { L"better" }, { L"economic" }, { L"effort" }, { L"up" }, { L"decide" }, { L"rate" }, { L"strong" }, { L"possible" }, { L"heart" }, { L"drug" },
        { L"show" }, { L"leader" }, { L"light" }, { L"voice" }, { L"wife" }, { L"whole" }, { L"police" }, { L"mind" }, { L"finally" }, { L"pull" }, { L"return" }, { L"free" },
        { L"military" }, { L"price" }, { L"report" }, { L"less" }, { L"according" }, { L"decision" }, { L"explain" }, { L"son" }, { L"hope" }, { L"even" }, { L"develop" },
        { L"view" }, { L"relationship" }, { L"carry" }, { L"town" }, { L"road" }, { L"drive" }, { L"arm" }, { L"true" }, { L"federal" }, { L"break" }, { L"better" },
        { L"difference" }, { L"thank" }, { L"receive" }, { L"value" }, { L"international " }, { L"building" }, { L"action" }, { L"full" }, { L"model" }, { L"join" },
        { L"season" }, { L"society" }, { L"because" }, { L"tax" }, { L"director" }, { L"early" }, { L"position " }, { L"player" }, { L"agree" }, { L"especially" },
        { L"record " }, { L"pick" }, { L"wear " }, { L"paper" }, { L"special" }, { L"space" }, { L"ground " }, { L"form" }, { L"support " }, { L"event" }, { L"official" },
        { L"whose " }, { L"matter" }, { L"everyone " }, { L"center" }, { L"couple" }, { L"site" }, { L"end" }, { L"project" }, { L"hit " }, { L"base" }, { L"activity" },
        { L"star" }, { L"table " }, { L"need " }, { L"court" }, { L"produce " }, { L"eat" }, { L"American" }, { L"teach" }, { L"oil " }, { L"half" }, { L"situation" }, { L"easy" },
        { L"cost" }, { L"industry" }, { L"figure" }, { L"face " }, { L"street " }, { L"image" }, { L"itself " }, { L"phone " }, { L"either" }, { L"data" }, { L"cover " },
        { L"quite" }, { L"picture" }, { L"clear" }, { L"practice" }, { L"piece" }, { L"land" }, { L"recent" }, { L"describe" }, { L"product" }, { L"doctor" }, { L"wall" },
        { L"patient " },{ L"worker" }, { L"news " }, { L"testtesttest" }, { L"movie" }, { L"certain " }, { L"north" }, { L"love" }, { L"personal " }, { L"open" }, { L"support" }, { L"simply" },
        { L"third" }, { L"technology" }, { L"catch" }, { L"step" }, { L"baby" }, { L"computer" }, { L"type " }, { L"attention" }, { L"draw" }, { L"film" }, { L"Republican" },
        { L"tree" }, { L"source" }, { L"red" }, { L"nearly" }, { L"organization" }, { L"choose" }, { L"cause" }, { L"hair" }, { L"look" },
        { L"point" }, { L"century" }, { L"evidence" }, { L"window " }, { L"difficult" },{ L"listen" }, { L"soon " }, { L"culture " }, { L"billion " }, { L"chance" }, { L"brother" }, { L"energy" },
        { L"period" }, { L"course " }, { L"summer" }, { L"less" }, { L"realize" }, { L"hundred" }, { L"available" }, { L"plant" }, { L"likely" }, { L"opportunity" },
        { L"term " }, { L"short " }, { L"letter" }, { L"condition" }, { L"choice" }, { L"place" }, { L"single" }, { L"rule" }, { L"daughter" }, { L"administration" },
        { L"south" }, { L"husband" }, { L"Congress" }, { L"floor" }, { L"campaign" }, { L"material" }, { L"population" }, { L"well" }, { L"call" }, { L"economy" },
        { L"medical" },{ L"hospital" }, { L"church" }, { L"close" },{ L"thousand" }, { L"risk" }, { L"current" }, { L"fire" }, { L"future" },
        { L"wrong" }, { L"involve" }, { L"defense" }, { L"anyone" }, { L"increase" }, { L"security" }, { L"bank" }, { L"myself" }, { L"certainly" }, { L"west" }, { L"sport" },
        { L"board" }, { L"seek" }, { L"per" }, { L"subject" }, { L"officer" }, { L"private" }, { L"rest" }, { L"behavior" }, { L"deal" }, { L"performance" }, { L"fight" },
        { L"throw" }, { L"top" }, { L"quickly" }, { L"past" }, { L"goal" }, { L"second" }, { L"bed" }, { L"order" }, { L"author" }, { L"fill" }, { L"represent" }, { L"focus" },
        { L"foreign" }, { L"drop" }, { L"plan" }, { L"blood" }, { L"upon" }, { L"agency" }, { L"push" }, { L"nature" }, { L"color" },
        { L"no" }, { L"recently" }, { L"store" }, { L"reduce" }, { L"sound" }, { L"note" }, { L"fine" }, { L"before" }, { L"near" }, { L"movement" }, { L"page" }, { L"enter" },
        { L"share" }, { L"than" }, { L"common" }, { L"poor" }, { L"other " }, { L"natural" }, { L"race" }, { L"concern" }, { L"series" }, { L"significant" }, { L"similar" },
        { L"hot" }, { L"language" }, { L"each" }, { L"usually" }, { L"response" }, { L"dead" }, { L"rise" }, { L"animal" }, { L"factor" }, { L"decade" }, { L"article" },
        { L"shoot" }, { L"east" }, { L"save" }, { L"seven" }, { L"artist" }, { L"away" }, { L"scene" }, { L"stock" }, { L"career" }, { L"despite" }, { L"central" }, { L"eight" },
        { L"thus" }, { L"treatment" }, { L"beyond" }, { L"happy" }, { L"exactly" }, { L"protect" }, { L"approach" }, { L"lie" }, { L"size" },
        { L"dog" }, { L"fund" }, { L"serious" },
        { L"occur" }, { L"media" }, { L"ready" }, { L"sign" }, { L"thought" }, { L"list" }, { L"individual" }, { L"simple" }, { L"quality" }, { L"pressure" }, { L"accept" },
        { L"answer" }, { L"hard" }, { L"resource" }, { L"identify" }, { L"left" }, { L"meeting" }, { L"determine" }, { L"prepare" }, { L"disease" }, { L"whatever" },
        { L"success" }, { L"argue" }, { L"cup" }, { L"particularly" }, { L"amount" }, { L"ability" }, { L"staff" }, { L"recognize" }, { L"indicate" }, { L"character" },
        { L"growth" }, { L"loss" }, { L"degree" }, { L"wonder" }, { L"attack" }, { L"herself" }, { L"region" }, { L"television" }, { L"box" }, { L"TV" }, { L"training" },
        { L"pretty" }, { L"trade" }, { L"deal" }, { L"election" }, { L"everybody" }, { L"physical" }, { L"lay" }, { L"general" }, { L"feeling" }, { L"standard" }, { L"bill" },
        { L"message" }, { L"fail" }, { L"outside" }, { L"arrive" }, { L"analysis" }, { L"benefit" }, { L"name" }, { L"sex" }, { L"forward" }, { L"lawyer" }, { L"present" },
        { L"section" }, { L"environmental" }, { L"glass" }, { L"answer" }, { L"skill" }, { L"sister" }, { L"PM" }, { L"professor" }, { L"operation" }, { L"financial" },
        { L"crime" }, { L"stage" }, { L"ok" }, { L"compare" }, { L"authority" }, { L"miss" }, { L"design" }, { L"sort" }, { L"one" }, { L"act" }, { L"ten" }, { L"knowledge" },
        { L"gun" }, { L"station" }, { L"blue" }, { L"state" }, { L"strategy" }, { L"little" }, { L"clearly" }, { L"discuss" }, { L"indeed" }, { L"force" }, { L"truth" },
        { L"song" }, { L"example" }, { L"democratic" }, { L"check" }, { L"environment" }, { L"leg" }, { L"dark" }, { L"public" }, { L"various" }, { L"rather" }, { L"laugh" },
        { L"guess" }, { L"executive" }, { L"set" }, { L"study" }, { L"prove" }, { L"hang" }, { L"entire" }, { L"rock" }, { L"design" }, { L"enough" }, { L"forget" }, { L"since" },
        { L"claim" }, { L"note" }, { L"remove" }, { L"manager" }, { L"help" }, { L"close" }, { L"sound" }, { L"enjoy" }, { L"network" }, { L"legal" }, { L"religious" },
        { L"cold" }, { L"form" }, { L"final" }, { L"main" }, { L"science" }, { L"green" }, { L"memory" }, { L"card" }, { L"above" }, { L"seat" }, { L"cell" }, { L"establish" },
        { L"nice" }, { L"trial" }, { L"expert" }, { L"that" }, { L"spring" }, { L"firm" }, { L"Democrat" }, { L"radio" }, { L"visit" }, { L"management" }, { L"care" },
        { L"avoid" }, { L"imagine" }, { L"tonight" }, { L"huge" }, { L"ball" }, { L"no" }, { L"close" }, { L"finish" }, { L"yourself" }, { L"talk" }, { L"theory" }, { L"impact" },
        { L"respond" }, { L"statement" }, { L"maintain" }, { L"charge" }, { L"popular" }, { L"traditional" }, { L"onto" }, { L"reveal" }, { L"direction" }, { L"weapon" },
        { L"employee" }, { L"cultural" }, { L"contain" }, { L"peace" }, { L"head" }, { L"control" }, { L"base" }, { L"pain" }, { L"apply" }, { L"play" }, { L"measure" },
        { L"wide" }, { L"shake" }, { L"fly" }, { L"interview" }, { L"manage" }, { L"chair" }, { L"fish" }, { L"particular" }, { L"camera" }, { L"structure" }, { L"politics" },
        { L"perform" }, { L"bit" }, { L"weight" }, { L"suddenly" }, { L"discover" }, { L"candidate" }, { L"top" }, { L"production" }, { L"treat" }, { L"trip" },
        { L"evening" }, { L"affect" }, { L"inside" }, { L"conference" }, { L"unit" }, { L"best" }, { L"style" }, { L"adult" }, { L"worry" }, { L"range" },
        { L"mention" }, { L"rather" }, { L"far" }, { L"deep" }, { L"front" }, { L"edge" }, { L"individual" }, { L"specific" }, { L"writer" }, { L"trouble" },
        { L"necessary" }, { L"throughout" }, { L"challenge" }, { L"fear" }, { L"shoulder" }, { L"institution" }, { L"middle" }, { L"sea" }, { L"dream" }, { L"bar" },
        { L"beautiful" }, { L"property" }, { L"instead" }, { L"improve" }, { L"stuff" }, { L"claim" },
};

size_t Dictionary::size() {
    return sizeof(dictionary) / sizeof(Word);
}

Word Dictionary::getRandom() {
    static std::random_device dev;
    static std::mt19937 rng(dev());
    static std::uniform_int_distribution<std::mt19937::result_type> dist6(0, size() - 1);

    const auto index = dist6(rng);
    return dictionary[index];
}

Word Dictionary::get(unsigned int index) {
    if(index > size())
        throw std::invalid_argument("Index is out of range.");
    return dictionary[index];
}
