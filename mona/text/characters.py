import random
from ..config import config

# Taveller is not included
characters_name_genshin = [
    "珐露珊",
    "流浪者",
    "纳西妲",
    "莱依拉",
    "赛诺",
    "坎蒂丝",
    "妮露",
    "柯莱",
    "多莉",
    "提纳里",
    "久岐忍",
    "鹿野院平藏",
    "夜兰",
    "瑶瑶",
    "神里绫人",
    "云堇",
    "八重神子",
    "申鹤",
    "荒泷一斗",
    "五郎",
    "托马",
    "埃洛伊",
    "珊瑚宫心海",
    "雷电将军",
    "九条裟罗",
    "宵宫",
    "早柚",
    "神里绫华",
    "枫原万叶",
    "优菈",
    "烟绯",
    "罗莎莉亚",
    "胡桃",
    "魈",
    "甘雨",
    "阿贝多",
    "钟离",
    "辛焱",
    "达达利亚",
    "迪奥娜",
    "可莉",
    "温迪",
    "刻晴",
    "莫娜",
    "七七",
    "迪卢克",
    "琴",
    "砂糖",
    "重云",
    "诺艾尔",
    "班尼特",
    "菲谢尔",
    "凝光",
    "行秋",
    "北斗",
    "香菱",
    "雷泽",
    "芭芭拉",
    "丽莎",
    "凯亚",
    "安柏",
    "白术",
    "卡维",
    "瑶瑶",
    "艾尔海森",
    "迪希雅",
    "米卡",
    "琳妮特",
    "林尼",
    "菲米尼",
    "芙宁娜",
    "那维莱特",
    "夏沃蕾",
    "娜维娅",
    "嘉明",
    "闲云",
    "千织",
    "阿蕾奇诺",
    "夏洛蒂",
    "莱欧斯利",
]

# Trailblazer is not included
# https://github.com/Mar-7th/StarRailRes/blob/master/index_new/cn/characters.json
characters_name_starrail = [
    "三月七",
    "丹恒",
    "姬子",
    "瓦尔特",
    "卡芙卡",
    "银狼",
    "阿兰",
    "艾丝妲",
    "黑塔",
    "布洛妮娅",
    "希儿",
    "希露瓦",
    "杰帕德",
    "娜塔莎",
    "佩拉",
    "克拉拉",
    "桑博",
    "虎克",
    "玲可",
    "卢卡",
    "托帕&账账",
    "青雀",
    "停云",
    "罗刹",
    "景元",
    "刃",
    "素裳",
    "驭空",
    "符玄",
    "彦卿",
    "桂乃芬",
    "白露",
    "镜流",
    "丹恒•饮月",
    "雪衣",
    "寒鸦",
    "藿藿",
    "加拉赫",
    "银枝",
    "阮•梅",
    "砂金",
    "真理医生",
    "花火",
    "黑天鹅",
    "黄泉",
    "知更鸟",
    "流萤",
    "米沙",
    "翡翠",
    "波提欧",
]

def random_equip():
    if config["model_type"] == "Genshin":
        return random.choice(characters_name_genshin) + "已装备"
    elif config["model_type"] == "StarRail":
        return "装备中"
