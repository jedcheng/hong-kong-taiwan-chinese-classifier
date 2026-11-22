"""Curated HK-form / TW-form pairs used only to VALIDATE the data-driven lexicon
(counted as raw substrings, so independent of jieba). (category, HK term, TW term)."""
PAIRS = [
 ("tech", "軟件", "軟體"), ("tech", "硬件", "硬體"), ("tech", "網上", "線上"), ("tech", "網絡", "網路"),
 ("tech", "互聯網", "網際網路"), ("tech", "人工智能", "人工智慧"), ("tech", "數碼", "數位"),
 ("tech", "上載", "上傳"), ("tech", "打印", "列印"), ("tech", "解像度", "解析度"), ("tech", "短訊", "簡訊"),
 ("tech", "內存", "記憶體"), ("tech", "鐳射", "雷射"), ("tech", "視像", "視訊"), ("tech", "默認", "預設"),
 ("tech", "手提電話", "手機"), ("tech", "屏幕", "螢幕"), ("tech", "私隱", "隱私"),
 ("transport", "的士", "計程車"), ("transport", "巴士", "公車"), ("transport", "港鐵", "捷運"),
 ("transport", "單車", "腳踏車"), ("transport", "泊車", "停車"), ("transport", "升降機", "電梯"),
 ("food", "菠蘿", "鳳梨"), ("food", "薯仔", "馬鈴薯"), ("food", "士多啤梨", "草莓"), ("food", "車厘子", "櫻桃"),
 ("food", "雪糕", "冰淇淋"), ("food", "朱古力", "巧克力"), ("food", "三文魚", "鮭魚"), ("food", "芝士", "起司"),
 ("food", "多士", "吐司"), ("food", "沙律", "沙拉"), ("food", "意粉", "義大利麵"), ("food", "外賣", "外送"),
 ("food", "飲管", "吸管"), ("food", "雪櫃", "冰箱"),
 ("daily", "紙巾", "面紙"), ("daily", "廁紙", "衛生紙"), ("daily", "膠袋", "塑膠袋"), ("daily", "質素", "品質"),
 ("daily", "渠道", "管道"), ("daily", "按揭", "房貸"), ("daily", "發展商", "建商"), ("daily", "樓盤", "建案"),
 ("daily", "落單", "下單"), ("daily", "健身室", "健身房"), ("daily", "幼稚園", "幼兒園"), ("daily", "功課", "作業"),
 ("daily", "課室", "教室"), ("daily", "保安", "警衛"),
 ("names", "意大利", "義大利"), ("names", "新西蘭", "紐西蘭"), ("names", "悉尼", "雪梨"),
 ("names", "特朗普", "川普"), ("names", "奧巴馬", "歐巴馬"),
 ("grammar", "嘅", "的"), ("grammar", "係", "是"), ("grammar", "唔", "不"), ("grammar", "佢", "他"),
]
