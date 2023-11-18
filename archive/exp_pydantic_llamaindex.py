from pydantic import BaseModel
from typing import List
from llama_index.program import OpenAIPydanticProgram
from dotenv import load_dotenv
from llama_index.llms import OpenAI

load_dotenv()


class Transcript(BaseModel):
    """Data model for transcript."""

    start: float
    end: float
    desc: str
    # transcript: str


class Transcripts(BaseModel):
    """Data model for transcripts."""

    transcripts: List[Transcript]


prompt_template_str = """\
Given the Youtube Transcript below, read it till the very end, pick up the main points and create timestamps.
Use the transcript time format (float) to create the timestamps.
Generate timestamps from the beginning till the end of the transcript.
All transcript time should be covered in the timestamps.

Youtube Transcript:
{yt_transcript}
"""

yt_transcript = """
{0.0: 'ANDREW HUBERMAN: Welcome to\nthe Huberman Lab podcast, where', 2.458: 'we discuss science and science\nbased tools for everyday life.', 5.67: '[MUSIC PLAYING]', 9.01: "I'm Andrew Huberman\nand I'm a professor", 10.81: 'of neurobiology\nand ophthalmology', 12.79: 'at Stanford School of Medicine.', 14.44: 'Today is an Ask\nMe Anything or AMA', 17.32: 'episode, which is part of our\npremium subscriber content.', 20.77: 'Our premium channel\nwas launched in order', 22.6: 'to raise support for the\nstandard Huberman Lab podcast', 24.97: 'channel, which still comes\nout once a week every Monday,', 28.06: 'and of course, is\nzero cost to consumer.', 30.43: 'The premium channel\nis also designed', 32.2: 'to support exciting\nresearch being', 34.09: 'done at major universities\nlike Stanford and elsewhere.', 36.79: "Research that's\ndone on humans that", 38.53: 'should lead to protocols for\nmental health, physical health', 41.29: 'and performance in\nthe near future.', 43.752: "If you'd like to check out the\npremium channel subscription", 46.21: 'model, you can go to\nhubermanlab.com/premium', 49.18: 'and there you can subscribe\nfor $10 a month or $100 a year.', 52.46: 'We also have a lifetime\nsubscriber option.', 54.31: 'For those of you that are\nalready Huberman Lab podcast', 56.56: "premium subscribers and you're\nwatching and/or hearing this,", 59.68: 'please go to\nhubermanlab.com/premium', 62.62: 'and download the\npremium podcast feed.', 65.019: 'And for those of you\nthat are not already', 66.82: 'Huberman Lab Premium\npodcast subscribers,', 68.95: 'you will be able to hear\nthe first 15 minutes or so', 71.643: 'of this episode\nand hopefully that', 73.06: 'will allow you to\ndiscern whether or not', 74.727: 'you would like to become\na premium subscriber.', 76.69: "Without further ado, let's get\nto answering your questions.", 80.21: 'And as always, I will strive\nto be as clear as possible,', 83.62: 'as succinct as possible, and\nas thorough as possible while', 88.0: 'still answering as many\nquestions per AMA episode', 91.63: 'as I can without these sessions\nbecoming unreasonably long.', 94.863: 'I should also point out\nthat if you asked a question', 97.03: 'and it was not\nanswered this AMA,', 99.34: 'it may very well be\nanswered in the next AMA.', 102.46: 'So the first question,\nwhich had a lot of upvotes--', 105.347: 'meaning many people wanted\nthe answer to this question--', 107.68: 'came from Jackson Lipfert.', 109.48: 'And the question was about\nso-called ultradian rhythms.', 112.45: 'For those of you that are\nnot familiar with ultradian', 114.73: 'rhythms-- ultradian\nrhythms are any rhythms', 117.16: 'that are shorter than 24 hours.', 119.71: 'And typically when\npeople ask about', 121.81: 'or talk about\nultradian rhythms, they', 123.73: 'are referring to\n90-minute rhythms.', 125.83: "I've talked about these\non the podcast before.", 127.81: "And Jackson's question was--\nhow do you use ultradian rhythms", 131.5: 'in your daily work?', 133.18: "There's more to the\nquestion, but first off, I", 135.64: 'do use ultradian\nrhythms-- that is,', 137.41: 'I leverage the fact that\nthese do exist in all of us', 141.37: 'as a way to engage in\nfocused bouts of mental work', 145.27: 'once or twice, or sometimes\nthree times per day.', 149.18: "However, I use them\nin a way that's", 151.54: 'grounded in the research on\nultradian rhythms for learning', 154.15: 'and memory in a\nway that might not', 155.952: 'be obvious just from\ntheir name that they', 157.66: 'are 90-minute rhythms.', 158.98: "So I'll get into the details\nof how to use ultradian rhythms", 162.52: 'to best capture\nneuroplasticity--', 164.528: "that is the brain's ability\nto change in response", 166.57: 'to experience-- and in a way\nthat should allow you to get', 170.47: 'one or two, or maybe even three\nfocused bouts of learning per', 174.7: 'day, which can greatly\naccelerate learning', 176.83: 'of cognitive material,\nlanguages, mathematics,', 180.22: 'history--', 181.06: 'for sake of school\nor work, or maybe', 182.71: 'just a hobby, or a personal\ninterest of some sort--', 185.02: 'and for skill learning in\nthe physical domain as well.', 187.9: 'Jackson then went on to ask--', 189.43: "you've mentioned\nbefore that you try", 190.93: 'to include at least one\n90-minute focus block per day', 193.572: 'as part of your work\nand overall mission.', 195.28: 'And indeed, that is true.', 196.42: 'I tried to get at least one of\nthese focused ultradian rhythm', 199.42: 'blocks per day.', 200.98: 'That is a period\nof about 90 minutes', 204.19: "where I'm focused on learning\nsomething or doing something", 206.71: "that's cognitively hard,\nalthough typically I", 209.17: 'aim for two of these\nsessions per day.', 211.51: 'He then goes on to ask-- what\nis the maximum number of blocks', 214.21: 'you can perform sustainably?', 216.16: 'The answer to that\nis probably four.', 219.4: 'And I say probably because\nsome people have schedules,', 222.46: 'lifestyles, in which\nfour 90-minute blocks', 226.45: 'of focused learning\nis possible per day,', 229.93: "but that's highly unusual.", 231.25: "For most people, it's going to\nbe one or two, maybe three--", 234.67: 'four, I would place in the\nreally extraordinary end', 237.13: "of things, maybe if\nyou're cramming for exams", 239.38: "or you've managed to go on a\nwriting retreat or a learning", 242.09: 'and retreat of some sort where\nyou can devote essentially,', 245.0: 'all of your non-sleeping,\nnon-eating time to learning.', 249.333: "But most people simply can't\norganize their life that way.", 251.75: 'So the short answer is--', 253.33: "for me, it's one or two\nper day is the target", 256.69: 'and three would be the maximum.', 259.06: 'You then went on to ask--\ndo you take vacations', 261.22: 'or extended breaks from these\nultradian rhythm sessions?', 264.55: 'And the short answer is no.', 267.13: 'Typically, I try and do this\nevery day-- and yes, even', 269.793: 'on the weekends.', 270.46: 'But on the weekend,\nthe ultradian rhythm', 272.77: 'focused learning\nbout might just be', 274.36: 'reading a book for\nabout 90 minutes', 276.4: 'or so, which might not be as\ncognitively difficult as it', 280.45: 'is for other sorts of work\nthat I perform during the week.', 282.97: 'I occasionally\nmiss a day entirely', 284.89: 'for whatever reason--\ntravel, obligations', 287.65: 'related to family, et cetera.', 289.57: 'But in general, I try\nand do this every day.', 291.94: 'I do think that the\ncircuits for focus are,', 295.36: 'I guess the non-biological way\nto put it would be kept warm.', 298.12: 'But essentially, that accessing\nthe circuits for focus', 301.36: 'is made easier by\naccessing them regularly', 304.182: "and that's because\nthe circuits for focus", 305.89: 'are indeed themselves\namenable to neuroplasticity.', 309.68: 'In other words, the more\nyou force yourself to focus,', 312.17: 'the easier focusing gets.', 314.015: "I'll now answer the last\npart of the question", 315.89: 'and then I will go through\nand emphasize some tools', 318.89: 'that anyone can use in order\nto leverage ultradian rhythms', 322.61: 'toward learning bouts--\neither cognitive learning', 325.52: 'or physical skill learning,\nor a combination of the two.', 328.978: 'The last part of the\nquestion Jackson asked', 330.77: 'was-- if you knew you needed\nto drastically increase', 333.41: 'the amount of focus you do\ndaily, how would you schedule', 336.11: 'that focus and recover from it?', 338.21: "That's an excellent\naspect to this question.", 341.27: 'And I will now give\nyou the details', 343.19: 'of how I would use and\nschedule ultradian rhythms.', 346.13: "I'll offer you a tool--", 347.15: "I've never talked about this\ntool in the Huberman Lab", 349.07: 'podcast.', 349.73: 'And I will dispel a common myth\nabout ultradian rhythms that', 353.3: 'points to a, believe it or\nnot, an easier way to leverage', 356.42: 'them for maximum benefit.', 358.18: 'OK.', 358.68: 'So as I mentioned before,\nultradian rhythms are', 360.71: 'these 90-minute cycles that\nwe go through from the time', 363.77: 'that we are born\nuntil the time we die.', 366.2: 'Indeed, even during\nsleep, we are', 368.99: 'experiencing and\nmore or less governed', 371.33: 'by these ultradian rhythms.', 373.25: 'This question and this answer\nis not so much about sleep.', 376.023: 'But just know that when you\ngo to sleep at night until you', 378.44: 'wake up in the morning,\nevery 90 minutes or so,', 380.84: 'your patterns of sleep-- that is\nthe percentage or ratio rather', 384.65: 'of slow wave sleep to light\nsleep to rapid eye movement', 387.41: 'sleep changes in a way such\nthat each 90-minute cycle gates', 392.905: 'the next cycle.', 393.53: 'It flips on switch for\nthe next 90-minute cycle', 396.41: 'then that 90-minute\ncycle ends, flips', 398.36: 'on switch for the next one,\nand so on and so forth.', 400.5: 'I mentioned all that\nbecause during the daytime,', 402.5: 'the same thing is true.', 404.42: "But most people don't know\nwhen the 90-minute all trading", 407.72: 'cycles begin.', 408.793: 'Because if you\nthink about it, you', 410.21: 'could wake up on the\nbasis of an alarm clock', 413.81: 'or noise in the room, or\nsimply because you naturally', 416.72: 'wake up in the middle of an\nultradian 90-minute cycle.', 420.21: 'So does that mean, for instance,\nthat if you wake up 60 minutes', 424.28: 'into an ultradian cycle\nthat the next 30 minutes', 427.43: 'of your waking--', 428.6: 'right, because that\n60 minutes needs', 430.31: 'to continue to 90 to\ncomplete an ultradian', 432.17: 'cycle-- that the next\n30 minutes after waking', 435.65: 'are related to the\nultradian cycle', 438.47: 'that you were still in\nduring sleep, or does', 441.5: 'it start a new ultradian cycle.', 443.81: 'And the answer is the former.', 445.31: 'That ultradian cycle\ncontinues even if you wake up', 448.25: 'in the middle of it.', 449.52: 'And so a lot of people who want\nto leverage ultradian cycles', 452.133: 'for learning will say, "Well,\nhow do I know when to start?', 454.55: 'When does it start,\nwhen I hit my stopwatch?', 456.44: 'Can I just set a clock\nand work for 90 minutes?"', 458.75: 'And the short answer is no.', 460.62: 'And that might seem\nunfortunate, but the good news', 463.28: 'is that you can figure out when\nyour first proper ultradian', 468.14: 'cycle of the day begins simply\nby asking yourself when are', 471.95: 'you most alert after waking.', 474.53: 'That is if you were, say\nto wake up at 7:00 AM', 477.23: "and let's say that's the end of\nan ultradian cycle or perhaps", 480.71: "you're in the middle\nof an ultradian cycle--", 482.58: "doesn't matter.", 484.07: 'What you need to watch\nfor or pay attention to', 486.35: 'for a day or so\nis when you start', 488.24: 'to experience your greatest\nstate of mental alertness', 492.38: 'in the morning.', 493.2: 'And here, we can discard with\nall the issues and variables', 495.68: 'around caffeine or no caffeine,\nhydrating or no hydrating.', 499.13: "Exercise is one variable that\nwe'll consider in a moment.", 502.62: "But here's the deal--", 504.56: 'these all trading in cycles\nare actually triggered', 507.56: 'by fluctuations in the so-called\nglucocorticoid system--', 512.36: 'the system that regulates\ncortisol release.', 515.12: 'And as some of you have\nprobably heard me say before--', 518.48: "cortisol, even though it's often\ndiscussed as a terrible thing,", 521.6: "it's chronic stress, cortisol,\ncortisol, et cetera-- cortisol", 525.02: 'is essential for\nhealth and every day we', 527.45: 'get a rise in cortisol in the\nmorning that is associated', 531.02: 'with enhanced immune function,\nenhanced alertness, enhanced', 534.02: 'ability to focus,\nso on and so forth.', 536.37: "In fact, the protocol\nthat I'm always", 538.033: 'beating the drum\nabout that people', 539.45: 'should get sunlight\nin their eyes', 541.1: 'as close to waking as possible--\nthat actually enhances', 543.83: "or increases the peak\nlevel of cortisol that's", 546.71: 'experienced early in the day.', 548.54: 'And that sets in motion a number\nof these ultradian cycles.', 551.34: 'So for instance, if\nyou wake up at 7:00 AM', 553.28: 'and you find that for the\nfirst hour after waking,', 555.86: 'you tend to be a\nlittle bit groggy,', 557.99: 'or you happen to be\ngroggy on a given day,', 560.18: 'but then you notice that your\nattention and alertness starts', 563.06: 'to peak somewhere around\n9:30 AM or 10:00 AM,', 567.17: 'you can be pretty sure\nthat, that first ultradian', 571.04: 'cycle for learning is going\nto be optimal to start', 574.49: 'at about 9:30 or 10:00 AM.', 576.65: "How can I say about if it's\nindeed a 90-minute cycle?", 580.34: 'Well, this is really where\nthe underlying neurobiology', 583.853: 'in these ultradian cycles\nconverge to give you', 585.77: 'a specific protocol.', 587.0: 'The changes in cortisol that\noccur throughout the day', 589.4: 'involve--', 589.94: "yes, a big peak early\nin the day if you're", 591.69: 'getting your sunlight and\ncaffeine and maybe even some', 594.14: 'exercise early in the day.', 595.73: 'But typically, that\npeak comes early.', 598.04: 'And then across the day the\nbaseline jitters a little bit,', 601.13: 'it comes down, but it\nbounces around a little bit.', 603.23: "It's not a flat line,\nif we were to measure", 605.09: 'your glucocorticoid levels.', 607.38: 'Each one of those little\nbumps corresponds to a shift', 610.16: 'in these ultradian cycles.', 612.21: 'So if you find that you\nare most alert at 9:30', 616.31: 'or starting to\nbecome alert at 9:30,', 618.32: 'and then typically you have a\npeak of focus and concentration', 621.05: 'around 10:00 AM, that is\nreally valuable to know.', 625.07: 'Because the way that\nthe molecules that', 628.1: 'control neuroplasticity--\nthat is the changes in neurons', 631.97: 'and other cell types\nin the brain that', 633.68: 'allow your nervous system\nto learn and literally', 636.44: 'for new connections to\nform between neurons, which', 638.9: 'is basically the\nbasis of learning--', 640.67: 'those fluctuate according\nto these ultradian cycles.', 644.57: 'What does this mean?', 645.69: 'This means if your peak\nin alertness and focus', 648.89: 'and energy--', 650.0: 'could even be experienced\nas physical energy occurs', 652.22: 'at about 9:30 AM, I would start\nyour first ultradian cycle', 656.06: 'for learning somewhere\naround there.', 657.92: 'Certainly 9:30 AM\nwould be ideal,', 659.99: 'but 10:00 AM would\nbe fine as well.', 661.86: 'And then you have\nabout one hour to get', 664.58: 'the maximum amount\nof learning in even', 666.56: 'within that alternating cycle.', 668.1: "This is where there's a\nlot of confusion out there,", 669.62: 'people think, oh, ultradian\ncycles are 90 minutes,', 672.0: 'therefore, we should be\nin our peak level of focus', 675.5: 'throughout that 90 minutes.', 676.73: 'In reality, most people\ntake about 10 or 15 minutes', 679.28: 'to break into a really\ndeep trench of focus', 682.1: 'and then periodically\nthroughout the next hour,', 684.62: "they'll pop out\nof that focus, now", 686.81: 'have to deliberately refocus.', 688.43: 'This is why, if\npossible, you want', 690.38: 'to turn off Wi-Fi\nconnections and put', 691.967: 'your phone in the other\nroom or turn it off.', 693.8: 'If.', 694.1: 'You do need your\nphone or Wi-Fi, just', 695.643: 'be aware of how\ndistracting those things', 697.31: 'can be to getting into\na deep trench of focus.', 700.37: 'But the point is this--', 702.5: 'these 90-minute cycles\noccur periodically', 704.69: 'throughout the day,\nbut there is going', 706.34: 'to be one period\nearly in the day--', 707.988: "and here, I'm referring\nto the spirit of starting", 710.03: 'at about 9:30 or 10:00\nAM-- and then likely', 712.55: 'another one in the mid to late\nafternoon that are going to be', 716.3: 'ideal for focused learning.', 719.15: 'And that focus learning\nbout should ideally', 722.81: 'have you set your clock--', 724.76: 'a stopwatch or something--\nto measure 90 minutes, but do', 728.48: "assume that there's going to\nbe some jitter at the front end", 731.32: "where you're not going\nto be able to focus", 733.07: 'as deeply as you would like.', 734.237: "Then you'll get about\nan hour of deep focus", 736.738: 'and then you really\nstart to transition out', 738.53: 'of these ultradian cycles.', 740.58: 'How do when the afternoon\nultradian cycle occurs?', 743.3: 'Well, just as in the\nmorning, it occurs', 745.55: "because there's a brief,\nbut significant increase", 748.61: 'in the glucocorticoid system\nin the mid to late afternoon.', 752.52: "I wish I could tell you\nit's going to be 2 PM", 754.58: "or it's going to be 3:00\nPM-- that's really going", 758.33: 'to depend on the individual.', 759.598: 'When you ingest caffeine,\nsome of the other', 761.39: 'demands of your day.', 762.71: 'But you can learn\nto recognize when', 765.41: 'these two periods for\noptimized learning', 768.2: 'will occur and here\nare the key principles.', 770.63: 'Watch for a day or two--\nmeaning pay attention', 772.79: 'to when you have your peak\nlevels of physical and mental', 775.22: 'energy in the morning-- that\nis between waking and noon,', 778.25: 'and then again between noon\nand about 6:00 or 7:00 PM.', 782.248: "Although I'm sure that there\nare some late shifted folks that", 784.79: 'will experience their peak in\nfocus somewhere around 6:00', 787.01: "or 7:00 PM, especially\nif they're waking up", 788.802: 'around 10:00 or 11:00 AM, as I\nknow some people out there are.', 792.23: 'Once you know where\nthose peaks in focus', 794.69: 'occur on your schedule,\nset a stopwatch', 798.5: 'for one ultradian cycle in\nthe early part of the day.', 801.53: 'In this example,\nI was saying 9:30,', 803.33: "but if you can't hop on\nit until 10, that's fine.", 806.0: 'Set it for 90 minutes,\nconsider that block wholly--', 808.88: 'meaning rule out all\nother distractions.', 810.86: 'But assume that within\nthat 90-minute block,', 813.08: 'you are only going to be\nable to focus intensely', 815.54: 'for about one hour.', 817.43: 'And just know that the molecules\nthat control neuroplasticity--', 820.613: 'and these things have names.', 821.78: 'And yes, Brain Derived\nNeurotrophic Factor or BDNF', 824.75: 'is the most famous of those,\nbut there are others as well.', 827.7: 'In fact, the very\nreceptors that control', 829.94: 'synaptic strength, the\nconnections between neurons,', 832.41: 'some of the neurotransmitters\nand modulators involved', 834.98: 'in synaptic plasticity,\nthey undergo regulation', 837.74: 'by these ultradian\nchanges in glucocorticoid.', 841.76: 'And then try and capture\na second ultradian', 845.51: 'learning block in the afternoon.', 847.52: 'Again, just knowing that\nthe first 10 or 15 minutes,', 849.74: 'consider it mental\nwarm up, and then you', 851.407: "get about an hour-- it's not\nexactly 60 minutes, but about", 854.18: 'an hour to maximize learning.', 855.525: "So if you're trying\nto learn something,", 857.15: 'really capture it during\nthat phase as well.', 859.26: 'Now, is there a\nthird opportunity', 861.41: 'or a fourth opportunity?', 862.91: "This relates to Jackson's\nquestion directly.", 865.46: 'And the short answer\nis not really,', 867.65: "unless you're somebody who\nrequires very little sleep.", 870.17: 'Within the 12 or\n16 hours that one', 872.42: 'tends to be awake during\nthe day or 18 hours', 874.67: 'that one tends to be awake,\nthere are really only two', 877.13: 'of these major peaks in the\nglucocorticoid system that', 879.71: 'trigger the onset of\nthe circadian cycles.', 881.72: "Again, there's a ramping\nup and a ramping down", 885.53: 'of glucocorticoids\nthroughout the day.', 887.28: 'But the real key\nhere is to learn', 888.68: 'when you tend to be\nmost focused based', 891.11: 'on your regular sleep,\nwake cycle, caffeine', 893.18: 'intake, exercise, et cetera.', 894.71: "And again, that's going to\nvary from person to person.", 896.99: 'And you really only have two\nopportunities or two ultradian', 899.66: 'cycles to capture in order\nto get the maximum focus', 903.41: 'challenging work done a.k.a.', 905.93: 'learning.', 906.66: 'So for somebody\nthat wants to learn', 908.3: 'an immense amount of material\nor who has the opportunity', 911.03: 'to capture another\nultradian cycle,', 914.99: 'the other time where that tends\nto occur is also early days.', 918.81: 'So some people by\nwaking up early', 921.02: 'and using stimulants like\ncaffeine and hydration', 923.84: 'or some brief\nhigh-intensity exercise,', 925.94: 'can trigger that cortisol\npulse to shift a little bit', 929.21: 'earlier so that they can\ncapture a morning work', 931.55: "block that occurs\nsomewhere, let's", 933.5: 'say between 6:00 and 7:30 AM.', 935.48: "So let's think about\nour typical person--", 937.61: "at least in my example that's\nwaking up around 7:00 AM.", 941.03: 'And then I said, has their first\nultradian work cycle really', 944.09: 'flip on because that bump in\ncortisol around 9:30 or 10:00', 947.91: 'AM.', 948.41: 'If that person were say to set\ntheir alarm clock for 5:30 AM,', 953.39: "then get up, get some artificial\nlight-- if the sun isn't out,", 957.98: 'turn on bright or\nartificial lights,', 959.81: 'or if the sun happens to\nbe up that time of year,', 962.51: 'get some sunlight in your eyes.', 964.01: 'But irrespective\nof sunlight, were', 967.18: 'to get a little bit of\nbrief high intensity', 969.1: 'exercise maybe 10 or 15 minutes\nof skipping rope or even', 972.19: 'just Jumping Jacks or go\nout for a brief jog, what', 975.46: 'happens then is the cortisol\npulse starts to shift earlier.', 978.68: 'And so the next day\nand the following day', 980.602: "and so on and so forth--\nprovided they're still doing", 982.81: 'that exercise first thing and\nideally getting some light', 985.72: 'in their eyes as well--', 986.8: 'well then they\nhave an opportunity', 988.24: 'to capture an increase\nin cortisol that', 990.34: 'is now shifted from about\n7:00 AM to about 8:30 AM.', 994.212: 'So they can capture\nan hour of work there', 995.92: 'and then they will also\nstill be within that rising', 1000.03: 'phase of cortisol\nin the 9:30 to 10:00', 1002.775: 'AM block that lasts\nuntil about 11:30 or so.', 1007.2: 'They might have lunch.', 1008.19: 'Perhaps after lunch, they\ndo a Non-Sleep Deep Rest--', 1010.23: "maybe they, don't maybe you're\na napper or maybe you're not--", 1012.79: "doesn't really matter.", 1013.99: 'And then in the afternoon--', 1016.44: 'and I would suspect\nit would now be', 1018.36: 'in the earlier afternoon\nsometime around 2:00 or 2:30', 1021.3: 'would be typical.', 1022.56: 'Although again, that exact time\nwill vary person to person.', 1026.67: 'Then they would want to schedule\nanother 90-minute work block.', 1029.76: "So that's how you\ncan capture three.", 1031.349: 'Now you can start\nto see also why', 1033.18: 'capturing four\nultradian work blocks', 1035.43: 'would be exceedingly rare.', 1036.88: "It's just not typical\nthat people are awake", 1039.583: 'for that much of the day, you\nhave to sleep at some point.', 1042.0: "And I should mention that if\nyou're going to force yourself", 1043.74: 'to wake up earlier on\na consistent basis,', 1045.54: 'you probably should be\ntrying to get to sleep', 1047.722: 'a little bit earlier as well.', 1048.93: "Because it's not\njust the quality,", 1050.347: 'but the duration of\nquality sleep that really', 1052.855: 'matters for learning.', 1053.73: 'And I should also\nremind everybody', 1056.43: 'that the actual\nrewiring of neurons', 1059.07: 'does not occur during\nany focused work block,', 1061.32: 'it actually occurs during\ndeep sleep the following night', 1063.72: 'and the following night, and\nduring Non-Sleep Deep Rest.', 1066.1: 'This is why Non-Sleep\nDeep Rest can accelerate', 1068.91: "learning because it's\nin states of rest", 1070.96: 'that the actual\nconnections between neurons', 1072.9: 'strengthen or weaken\nor new neurons', 1075.06: 'are added in a way that allows\nfor what we call learning.', 1078.19: 'OK.', 1078.69: 'So one or two all trading work\nblocks per day is typical,', 1082.98: 'three would be\nreally exceptional,', 1085.41: 'and four would be extraordinary.', 1087.78: 'Look for them-- meaning look to\nsee when you are feeling most', 1092.28: 'focused and alert\ntypically in the period', 1095.16: 'before waking and\nnoon and typically', 1097.62: 'in the period between\nnoon and bedtime,', 1100.14: 'given your standard intake\nof caffeine and exercise', 1103.92: 'and other life events.', 1106.2: "Please also remember\nthat even though it's", 1109.14: 'an ultradian\n90-minute work block,', 1111.57: 'the neuroplasticity is\ngoing to be best triggered', 1114.3: 'within a 60-minute\nportion of that.', 1117.24: "And there's no way to\nknow exactly when that 60", 1119.79: 'minutes begins and\nends until you actually', 1122.185: 'begin the work block.', 1123.06: 'So this is really\ndesigned to be empirical,', 1125.177: 'you need to actually go do this.', 1126.51: "What you'll notice\nagain is that it's", 1127.83: "hard to focus at\nfirst then you'll", 1129.247: 'drop into a state of focus.', 1130.44: "You may get distracted,\nthat's perfectly normal.", 1132.54: 'You refocus, get\nback into triggering', 1134.118: "learning-- that's really\nwhat you're doing,", 1135.91: "you're triggering learning--\nand then there'll be some taper", 1138.21: "and then you'll be out of\nthe ultradian work block.", 1140.335: "Now, it's also a\nkey to understand", 1143.28: 'that myself and other\npeople should not', 1145.74: "expect that they're only working\nduring these 90-minute work", 1148.64: 'blocks.', 1149.14: "It's just that a lot of the\nsorts of demands of our day,", 1151.66: 'including cooking and shopping\nfor groceries, and email,', 1157.07: 'and text messaging, and social\nmedia-- a lot of those things', 1159.57: "don't require intense\nfocus of the sort", 1161.67: 'that I believe Jackson is\nasking about maximizing,', 1164.253: "and that I'm referring to when I\ntalk about these ultradian work", 1166.92: 'blocks.', 1167.52: "And then as a final\npoint, I've been", 1169.223: 'talking about these ultradian\nwork blocks and focus,', 1171.39: 'et cetera, in a context\nthat brings to mind ideas', 1174.75: 'about cognitive work.', 1176.16: 'So learning a language,\nlearning math, writing,', 1178.86: 'or creating, doing something\nrelated to music, et cetera.', 1181.74: 'But these 90-minute\nultradian work blocks', 1184.53: 'also directly relate to\nphysical skill learning as well', 1188.7: 'and to physical\nexercise as well.', 1190.57: "So if you are somebody who's\nreally interested in improving", 1193.56: 'your fitness and\nyour fitness requires', 1195.39: 'a lot of focused attention--', 1197.1: 'so for instance, when I go\nout for a long run on Sundays,', 1200.29: 'which is part of\nmy fitness routine,', 1202.29: 'I deliberately not\nthinking about much,', 1204.047: "I'm just trying to cruise along.", 1205.38: 'I might focus a\nlittle bit on my pace', 1206.922: "in stride, maybe an audiobook\nI'm listening to or a podcast.", 1210.3: "But typically, I'm\njust cruising along,", 1212.82: "it's low cognitive demand work.", 1214.98: 'These ultradian work\nblocks can really', 1216.78: 'be maximized for\npure cognitive work--', 1220.62: 'book type work, et cetera,\nmusic, et cetera, or they', 1223.89: 'can also be leveraged\ntoward skill learning.', 1226.81: "So if you're trying\nto learn how to dance", 1229.05: 'or how to perform a\nparticular athletic move', 1231.24: "or you're trying to get better\nat some skill that requires", 1234.09: 'a lot of focus and alignment of\nmuscular movement and cognitive', 1239.58: 'demands, et cetera,\nwell then these', 1241.92: 'are also going to be ideal\nfor triggering neuroplasticity', 1244.59: "to get better in the motor skill\nbased domain as it's called,", 1249.45: 'et cetera.', 1250.14: "If you'd like to learn\nmore about ultradian", 1252.69: 'shifts in neuroplasticity\nand ultradian work bouts,', 1256.02: 'I will certainly do more on this\nin the standard Huberman Lab', 1258.9: 'podcast.', 1259.71: 'But the key words\nto look up if you', 1261.918: 'want to explore this\nfurther online--', 1263.46: "it's not something that\na lot of people about,", 1265.74: "it's called iterative\nmetaplasticity.", 1268.25: "It's a vast literature\nand one that I'd", 1270.53: 'be happy to teach you in a\nstandard podcast episode.', 1272.9: 'But in the interest of getting\nto more questions from you', 1276.5: "all, hopefully the\nanswer I've given", 1278.93: 'you now has been complete\nenough, yet clear enough,', 1283.28: 'and yet succinct enough\nthat you can start', 1286.28: 'to leverage these\nreally powerful aspects', 1288.59: 'of iterative metaplasticity and\nultradian rhythms for learning.', 1292.46: "And I'd just like to point\nout that these opportunities", 1295.7: 'for focused learning that occur\nin these 90-minute ultradian', 1299.87: 'cycles are really\nterrific opportunities.', 1302.69: 'They are offered to you\nat least twice every day', 1305.78: 'and you can really learn\nto detect when they occur', 1308.66: "and when they're\nlikely to occur.", 1310.59: 'You can certainly learn at other\ntimes in the 24-hour cycle.', 1314.37: "But for anyone who's tried to\nstay up late at night cramming", 1317.42: "for an exam or\nfor somebody who's", 1319.34: 'tried to learn during\nthe sleepiest time', 1322.46: 'of their afternoon, we can be\nvery familiar with the fact', 1327.11: 'that there are times of day in\nwhich we are best at learning.', 1330.32: "And as I've just\ndescribed, there", 1332.39: 'are ways to capture\nthose moments', 1334.76: 'and they are valuable moments.', 1336.53: "So even though it's just\nabout three hours per day", 1339.187: 'or really only two hours\nper day because of the 60', 1341.27: 'to 90-minute thing that I talked\nabout a few minutes ago, learn', 1344.33: 'to know when these occur and\nreally treat them as valuable,', 1348.05: 'maybe even wholly in the sense\nthat they are really the times', 1351.23: 'that are offered\nup to you each day', 1352.79: 'by your own biology in ways\nthat will allow you to get', 1355.85: 'better pretty much at anything.', 1357.8: 'Thank you for joining\nfor the beginning', 1359.51: 'of this Ask Me Anything episode.', 1361.34: 'To hear the full episode and to\nhear future episodes of these', 1365.18: 'Ask Me Anything sessions, plus\nto receive transcripts of them', 1368.72: 'and transcripts of the Huberman\nLab podcast standard channel', 1371.87: 'and premium tools not\nreleased anywhere else,', 1374.87: 'please go to\nhubermanlab.com/premium.', 1377.785: 'Just to remind you why we\nlaunched the Huberman Lab', 1379.91: 'podcast premium channel--', 1381.41: "it's really twofold.", 1382.44: "First of all, it's\nto raise support", 1384.23: 'for the standard Huberman\nLab podcast channel, which', 1387.14: 'of course, will still be\ncontinued to be released', 1389.72: 'every Monday in full length.', 1391.328: 'We are not going to\nchange the format', 1392.87: 'or anything about the\nstandard Huberman Lab podcast.', 1396.35: 'And to fund research--\nin particular,', 1398.39: 'research done on human beings.', 1400.02: 'So not animal models,\nbut on human beings,', 1402.082: 'which I think we all\nagree is the species', 1403.79: 'that we are most interested in.', 1405.77: 'And we are going to\nspecifically fund', 1408.2: 'research that is aimed\ntoward developing', 1410.63: 'further protocols for mental\nhealth, physical health,', 1412.993: 'and performance.', 1413.66: 'And those protocols\nwill be distributed', 1415.52: 'through all channels.', 1416.397: 'Not just the premium channel,\nbut through all channels--', 1418.73: 'Huberman Lab podcast and\nother media channels.', 1421.05: 'So the idea here is to give\nyou information to your burning', 1424.13: 'questions in depth and\nallow you the opportunity', 1427.28: 'to support the research that\nprovides those kinds of answers', 1430.67: 'in the first place.', 1431.565: 'Now an especially exciting\nfeature of the premium channel', 1433.94: 'is that the Tiny\nFoundation has generously', 1436.46: 'offered to do a dollar\nfor dollar match', 1438.62: 'on all funds raised for research\nthrough the premium channel.', 1442.29: "So this is a terrific\nway that they're", 1444.02: 'going to amplify whatever funds\ncome in through the premium', 1446.478: 'channel to further support\nresearch for science', 1448.768: 'and science related tools for\nmental health, physical health,', 1451.31: 'and performance.', 1452.117: "If you'd like to sign up for the\nHuberman Lab Premium channel,", 1454.7: "again there's a cost\nof $10 per month", 1456.86: 'or you can pay $100 up\nfront for the entire year.', 1459.5: 'That will give you\naccess to all the AMAs', 1461.93: 'you can ask questions and\nget answers to your questions', 1464.69: "and you'll of course get\nanswers to all the questions", 1467.45: 'that other people ask as well.', 1468.74: 'There will also be some premium\ncontent such as transcripts', 1471.74: 'of the AMAs and various\ntranscripts and protocols', 1474.44: 'of Huberman Lab podcast episodes\nand not found elsewhere.', 1477.65: "And again, you'll be\nsupporting research", 1479.9: 'for mental health, physical\nhealth, and performance.', 1482.13: 'You can sign up for the\npremium channel by going', 1484.13: 'to hubermanlab.com/premium.', 1486.47: "Again, that's\nhubermanlab.com/premium.", 1489.26: 'And as always, thank you for\nyour interest in science.', 1493.09: '[MUSIC PLAYING]'}
"""

program = OpenAIPydanticProgram.from_defaults(
    llm=OpenAI(model="gpt-4-1106-preview", temperature=0.1),
    output_cls=Transcripts,
    prompt_template_str=prompt_template_str,
    verbose=True,
)

output = program(yt_transcript=yt_transcript)

# print(output)