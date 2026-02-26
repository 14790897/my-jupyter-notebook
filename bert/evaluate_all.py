"""
BERT完形填空测试评估脚本
自动运行所有测试并输出准确率表格
"""

from bert_utils import autoregressive_cloze_test, load_model

# 加载模型
print("=" * 80)
print("正在加载模型...")
print("=" * 80)
tokenizer, model, device = load_model("roberta-base")


# ============================================================================
# 测试1: 2022年考研英语一完形填空
# ============================================================================
print("\n" + "=" * 80)
print("测试1: 2022年考研英语一完形填空")
print("=" * 80)

raw_text_2022 = """
The idea that plants have some degree of consciousness first took root in the early 2000s; the term "plant neurobiology" was __1__ around the notion that some aspects of plant behavior could be __2__ to intelligence in animals. __3__ plants lack brains, the firing of electrical signals in their stems and leaves nonetheless triggered responses that __4__ consciousness, researchers previously reported.

But such an idea is untrue, according to a new opinion article. Plant biology is complex and fascinating, but it __5__ so greatly from that of animals that so-called __6__ of plants' intelligence is inconclusive, the authors wrote.

Beginning in 2006, some scientists have __7__ that plants possess neuron-like cells that interact with hormones and neurotransmitters, __8__ "a plant nervous system, __9__ to that in animals," said lead study author Lincoln Taiz, "They __10__ claimed that plants have "brain-like command centers" at their root tips."

This __11__ makes sense if you simplify the workings of a complex brain, __12__ it to an array of electrical pulses; cells in plants also communicate through electrical signals. __13__, the signaling in a plant is only __14__ similar to the firing in a complex animal brain, which is more than "a mass of cells that communicate by electricity," Taiz said.

"For consciousness to evolve, a brain with a threshold __15__ of complexity and capacity is required," he __16__."Since plants don't have nervous systems, the __17__ that they have consciousness are effectively zero."

And what's so great about consciousness, anyway? Plants can't run away from __18__, so investing energy in a body system which __19__ a threat and can feel pain would be a very __20__ evolutionary strategy, according to the article.
"""

options_dict_2022 = {
    1: ["coined", "discovered", "collected", "issued"],
    2: ["attributed", "directed", "compared", "confined"],
    3: ["unless", "when", "once", "though"],
    4: ["coped with", "consisted of", "hinted at", "extended"],
    5: ["suffers", "benefits", "develops", "differs"],
    6: ["acceptance", "evidence", "cultivation", "creation"],
    7: ["doubted", "denied", "argued", "requested"],
    8: ["adapting", "forming", "repairing", "testing"],
    9: ["analogous", "essential", "suitable", "sensitive"],
    10: ["just", "ever", "still", "even"],
    11: ["restriction", "experiment", "perspective", "demand"],
    12: ["attaching", "reducing", "returning", "exposing"],
    13: ["However", "Moreover", "Therefore", "Otherwise"],
    14: ["temporarily", "literally", "superficially", "imaginarily"],
    15: ["list", "level", "label", "local"],
    16: ["recalled", "agreed", "questioned", "added"],
    17: ["chances", "risks", "excuses", "assumptions"],
    18: ["danger", "failure", "warning", "control"],
    19: ["represents", "includes", "reveals", "recognizes"],
    20: ["humble", "poor", "practical", "easy"],
}

# 标准答案（基于选项索引：0=A, 1=B, 2=C, 3=D）
answers_2022 = {
    1: 0, 2: 2, 3: 3, 4: 2, 5: 3, 6: 1, 7: 2, 8: 1, 9: 0, 10: 3,
    11: 2, 12: 1, 13: 0, 14: 2, 15: 1, 16: 3, 17: 0, 18: 0, 19: 3, 20: 1
}

results_2022 = autoregressive_cloze_test(
    raw_text_2022, options_dict_2022, tokenizer, model, device,
    start_idx=1, end_idx=20
)


# ============================================================================
# 测试2: 2023年考研英语一完形填空
# ============================================================================
print("\n" + "=" * 80)
print("测试2: 2023年考研英语一完形填空")
print("=" * 80)

raw_text_2023 = """
Caravanserais were roadside inns that were built along the Silk Road in areas including China, North Africa and the Middle East. They were typically __1__ outside the walls of a city or village and were usually funded by governments or __2__. This word "Caravanserais" is a __3__ of the Persian word "karvan", which means a group of travellers or a caravan, and seray, a palace or enclosed building. The term caravan was used to __4__ groups of people who travelled together across the ancient network for safety reasons, __5__ merchants, travellers or pilgrims. From the 10th century onwards, as merchant and travel routes become more developed, the __6__ of the Caravanserais increased and they served as a safe place for people to rest at night. Travellers on the Silk Road __7__ the possibility of being attacked by thieves or being __8__ to extreme conditions. For this reason, Caravanserais were strategically placed __9__ they could be reached in a day's travel time. Caravanserais served as an informal __10__ point for the various people who travelled the Silk Road. __11__, those structures became important centers for culture __12__ and interaction, with travelers sharing their cultures, ideas and beliefs, __13__ taking knowledge with them, greatly __14__ the development of several civilizations. Caravanserais were also an important marketplace for commodities and __15__ in the trade of goods along the Silk Road. __16__, it was frequently the first stop for merchants looking to sell their wares and __17__ supplies for their own journeys. It is __18__ that around 12,000 to 15,000 caravanserais were built along the Silk Road, __19__ only about 3,000 are known to remain today, many of which are in __20__.
"""

options_dict_2023 = {
    1: ["displayed", "occupied", "located", "equipped"],
    2: ["privately", "regularly", "respectively", "permanently"],
    3: ["definition", "transition", "substitution", "combination"],
    4: ["classify", "record", "describe", "connect"],
    5: ["apart from", "instead of", "such as", "along with"],
    6: ["construction", "restoration", "impression", "evaluation"],
    7: ["doubted", "faced", "accepted", "reduced"],
    8: ["assigned", "subjected", "accustomed", "opposed"],
    9: ["so that", "even if", "now that", "in case"],
    10: ["talking", "starting", "breaking", "meeting"],
    11: ["By the way", "On occasion", "In comparison", "As a result"],
    12: ["heritage", "revival", "exchange", "status"],
    13: ["with regard to", "in spite of", "as well as", "in line with"],
    14: ["completing", "influencing", "resuming", "pioneering"],
    15: ["aided", "invested", "failed", "competed"],
    16: ["Rather", "Indeed", "Otherwise", "However"],
    17: ["go in for", "stand up for", "close in on", "stock up on"],
    18: ["believed", "predicted", "recalled", "implied"],
    19: ["until", "because", "unless", "although"],
    20: ["ruins", "debt", "fashion", "series"],
}

answers_2023 = {
    1: 2, 2: 0, 3: 3, 4: 2, 5: 2, 6: 0, 7: 1, 8: 1, 9: 0, 10: 3,
    11: 3, 12: 2, 13: 2, 14: 1, 15: 0, 16: 1, 17: 3, 18: 0, 19: 3, 20: 0
}

results_2023 = autoregressive_cloze_test(
    raw_text_2023, options_dict_2023, tokenizer, model, device,
    start_idx=1, end_idx=20
)


# ============================================================================
# 测试3: 2019年上海英语高考完形填空
# ============================================================================
print("\n" + "=" * 80)
print("测试3: 2019年上海英语高考完形填空")
print("=" * 80)

raw_text_2019_gaokao = """
We're told that writing is dying. Typing on keyboards and screens __1__ written communication today. Learning cursive, joined-up handwriting was once __2__ in schools. But now, not so much. Countries such as Finland have dropped joined-up handwriting lessons in __3__ of typing courses. And in the US, the requirement to learn cursive has been left out of core standards since 2013. A few US states still place value on formative cursive education, such as Arizona, but they're not the __4__.

Some experts point out that writing lessons can have indirect __5__. Anne Trubek, author of The History and Uncertain Future of Handwriting, argues that such lessons can reinforce a skill called automaticity. That's when you've perfected a task, and can do it almost without thinking, __6__ you extra mental bandwidth to think about other things while you're doing the task. In this sense, Trubek likens handwriting to __7__.

"Once you have driven for a while, you don't __8__ think 'Step on gas now' [or] 'Turn the steering wheel a bit'," she explains. "You just do it. That's what we want children to __9__ when learning to write. You don't think 'now make a loop going up for the 't'' or 'now look for the letter 'r' on the keyboard'."

Trubek has written many essays and books on handwriting, and she doesn't believe it will die out for a very long time, "ever", but she believes students are learning how to type faster without looking at the keys at __10__ ages, and students are learning automaticity with keyboards that was once exclusive to handwriting: to type faster than they could write, granting them extra time to think about word choice or sentence structure. In a piece penned for the New York Times last year, Trubek argued that due to the improved automaticity of keyboards, today's children may well become better communicators in text, as __11__ take up less of their education. 

This is a(n) __12__ that has attracted both criticism and support. She explains that two of the most common arguments she hears from detractors regarding the decline of handwriting is that not __13__ it will result in a loss of history and a "loss of personal touch".

On the former she __14__ that 95% of handwritten manuscripts can't be read by the average person anyway – "that's why we have paleographers," she explains, paleography being the study of ancient styles of writing – while the latter refers to the warm __15__ we give to handwritten personal notes, such as thank-you cards.
"""

options_dict_2019_gaokao = {
    1: ["abandons", "dominates", "enters", "absorbs"],
    2: ["compulsory", "opposite", "crucial", "relevant"],
    3: ["in want of", "in case of", "in favour of", "in addition to"],
    4: ["quantity", "minimum", "quality", "majority"],
    5: ["responsibility", "benefits", "resources", "structure"],
    6: ["granting", "getting", "bringing", "costing"],
    7: ["sleeping", "driving", "reviewing", "operating"],
    8: ["eventually", "constantly", "frequently", "consciously"],
    9: ["adopt", "reach", "acquire", "activate"],
    10: ["slower", "later", "faster", "earlier"],
    11: ["handwriting", "typing", "reading", "spelling"],
    12: ["trust", "book", "view", "smile"],
    13: ["containing", "spreading", "choosing", "preserving"],
    14: ["commits", "counters", "completes", "composes"],
    15: ["associations", "resources", "procedures", "interactions"],
}

answers_2019_gaokao = {
    1: 1, 2: 0, 3: 2, 4: 3, 5: 1, 6: 0, 7: 1, 8: 3, 9: 2, 10: 3,
    11: 0, 12: 2, 13: 3, 14: 1, 15: 0
}

results_2019_gaokao = autoregressive_cloze_test(
    raw_text_2019_gaokao, options_dict_2019_gaokao, tokenizer, model, device,
    start_idx=1, end_idx=15
)


# ============================================================================
# 测试4: 2019年上海英语春考完形填空
# ============================================================================
print("\n" + "=" * 80)
print("测试4: 2019年上海英语春考完形填空")
print("=" * 80)

raw_text_2019_spring = """
More people are travelling than ever before, and lower barriers to entry and falling costs means they are doing so for __41__ periods.

The rise of "city breaks" 48-hour bursts of foreign cultures, easier on the pocket and annual leave balance has increased tourist numbers, but not their __42__ spread. The same attractions have been used to market cities such as Paris, Barcelona and Venice for decades, and visitors use the same infrastructure as residents to reach them. "Too many people do the same thing at the exact same time," says Font. "For __43__, the city no longer belongs to them."

This starts with marketing, says Font, who notes that Amsterdam has started advising visitors to seek __44__ outside of the city center on its official website. "That takes some balls, really, to do that. But only so many people will look at the website, and it means they can say to their residents they're doing all they can (to ease congestion)."

But it also __45__ a better way, it is calling "de-tourism": sustainable travel tips and __46__ itineraries for exploring an authentic Venice, off the paths beaten by the 28 million visitors who flock there each year.

A greater variety of __47__ for prospective visitors—ideas for what to do in off-peak seasons, for example, or outside of the city center—can have the effect of diverting them from already saturated landmarks, or __48__ short breaks away in the first place.

Longer stays __49__ the pressure, says Font. "If you go to Paris for two days, you're not going to go to the Eiffel Tower. If you go for two weeks, you're not going to go to the Eiffel tower 14 times."

Similarly, repeat visitors have a better sense of the __50__, "We should be asking how do we get tourists to __51__, not how to get them to come for the first time. If they're coming for the fifth time, it is much easier to integrate their behavior with ours."

Local governments can foster this sustainable activity by giving preference to responsible operators and even high-paying consumers. Font says cities could stand to be more selective about the tourists they try to attract when the current metric for marketing success is how many there are, and how far they've come. "You're thinking, 'yeah but at what cost...'"

He points to unpublished data from the Barcelona Tourist Board that prioritizes Japanese tourists for spending an average of 640 more per day than French tourists—a(n) __52__ that fails to take into account their bigger carbon footprint. __53__ tourists are also more likely to be repeat visitors that come at off-peak times, buy local product, and __54__ less crowded parts of the city—all productive steps towards more __55__ and more peaceful relations with residents.
"""

options_dict_2019_spring = {
    41: ["longer", "shorter", "wider", "clearer"],
    42: ["environmental", "national", "economic", "geographic"],
    43: ["locals", "tourists", "visitors", "cleaners"],
    44: ["transports", "accommodation", "restaurants", "service"],
    45: ["addresses", "introduces", "proposes", "receives"],
    46: ["separate", "individual", "alternative", "objective"],
    47: ["reform", "guidance", "invitation", "support"],
    48: ["convincing", "discouraging", "preventing", "resisting"],
    49: ["peace", "risk", "leisure", "ease"],
    50: ["culture", "knowledge", "entertainment", "ability"],
    51: ["go with", "bring up", "come back", "lay off"],
    52: ["distinction", "harmony", "association", "comparison"],
    53: ["French", "Italian", "Spanish", "German"],
    54: ["carry out", "give into", "spread out", "impact on"],
    55: ["sight", "complex", "temporary", "sustainable"],
}

answers_2019_spring = {
    41: 1, 42: 3, 43: 0, 44: 1, 45: 2, 46: 2, 47: 1, 48: 1, 49: 3, 50: 0,
    51: 2, 52: 3, 53: 0, 54: 2, 55: 3
}

results_2019_spring = autoregressive_cloze_test(
    raw_text_2019_spring, options_dict_2019_spring, tokenizer, model, device,
    start_idx=41, end_idx=55
)


# ============================================================================
# 统计和输出结果
# ============================================================================
print("\n" + "=" * 80)
print("统计结果")
print("=" * 80)

def calculate_accuracy(results, answers, letters=["A", "B", "C", "D"]):
    """计算准确率"""
    correct = 0
    total = 0
    details = []
    
    for q_num in sorted(results.keys()):
        if q_num in answers:
            model_idx = letters.index(results[q_num]["letter"])
            answer_idx = answers[q_num]
            is_correct = (model_idx == answer_idx)
            
            if is_correct:
                correct += 1
            total += 1
            
            details.append({
                "q_num": q_num,
                "model_choice": f"{results[q_num]['letter']}. {results[q_num]['option']}",
                "answer": letters[answer_idx],
                "correct": is_correct
            })
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total, details

# 计算各项测试的准确率
letters = ["A", "B", "C", "D"]
acc_2022, correct_2022, total_2022, details_2022 = calculate_accuracy(results_2022, answers_2022, letters)
acc_2023, correct_2023, total_2023, details_2023 = calculate_accuracy(results_2023, answers_2023, letters)
acc_2019_gaokao, correct_2019_gaokao, total_2019_gaokao, details_2019_gaokao = calculate_accuracy(results_2019_gaokao, answers_2019_gaokao, letters)
acc_2019_spring, correct_2019_spring, total_2019_spring, details_2019_spring = calculate_accuracy(results_2019_spring, answers_2019_spring, letters)

# 输出汇总表格
print("\n" + "=" * 80)
print("准确率汇总表")
print("=" * 80)
print(f"{'测试名称':<30} {'正确数':<10} {'总题数':<10} {'准确率':<10}")
print("-" * 80)
print(f"{'2022年考研英语一':<30} {correct_2022:<10} {total_2022:<10} {acc_2022:>6.2f}%")
print(f"{'2023年考研英语一':<30} {correct_2023:<10} {total_2023:<10} {acc_2023:>6.2f}%")
print(f"{'2019年上海高考':<30} {correct_2019_gaokao:<10} {total_2019_gaokao:<10} {acc_2019_gaokao:>6.2f}%")
print(f"{'2019年上海春考':<30} {correct_2019_spring:<10} {total_2019_spring:<10} {acc_2019_spring:>6.2f}%")
print("-" * 80)
total_correct = correct_2022 + correct_2023 + correct_2019_gaokao + correct_2019_spring
total_questions = total_2022 + total_2023 + total_2019_gaokao + total_2019_spring
total_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
print(f"{'总计':<30} {total_correct:<10} {total_questions:<10} {total_accuracy:>6.2f}%")
print("=" * 80)

# 输出详细结果表格
print("\n" + "=" * 80)
print("详细结果 - 2022年考研英语一")
print("=" * 80)
print(f"{'题号':<6} {'模型选择':<20} {'标准答案':<10} {'结果':<6}")
print("-" * 80)
for detail in details_2022:
    mark = "✅" if detail["correct"] else "❌"
    print(f"{detail['q_num']:<6} {detail['model_choice']:<20} {detail['answer']:<10} {mark:<6}")

print("\n" + "=" * 80)
print("详细结果 - 2023年考研英语一")
print("=" * 80)
print(f"{'题号':<6} {'模型选择':<20} {'标准答案':<10} {'结果':<6}")
print("-" * 80)
for detail in details_2023:
    mark = "✅" if detail["correct"] else "❌"
    print(f"{detail['q_num']:<6} {detail['model_choice']:<20} {detail['answer']:<10} {mark:<6}")

print("\n" + "=" * 80)
print("详细结果 - 2019年上海高考")
print("=" * 80)
print(f"{'题号':<6} {'模型选择':<20} {'标准答案':<10} {'结果':<6}")
print("-" * 80)
for detail in details_2019_gaokao:
    mark = "✅" if detail["correct"] else "❌"
    print(f"{detail['q_num']:<6} {detail['model_choice']:<20} {detail['answer']:<10} {mark:<6}")

print("\n" + "=" * 80)
print("详细结果 - 2019年上海春考")
print("=" * 80)
print(f"{'题号':<6} {'模型选择':<30} {'标准答案':<10} {'结果':<6}")
print("-" * 80)
for detail in details_2019_spring:
    mark = "✅" if detail["correct"] else "❌"
    print(f"{detail['q_num']:<6} {detail['model_choice']:<30} {detail['answer']:<10} {mark:<6}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
