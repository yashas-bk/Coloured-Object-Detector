"""Default vocabulary for CLIP zero-shot naming of unrecognized detections.

A curated list of everyday objects, biased toward things that (a) come in
distinct colours and (b) are NOT in COCO's 80 classes — those are exactly the
objects that reach the CLIP naming stage, because YOLO already names what it
knows. Users can extend this per request with custom labels.
"""

DEFAULT_VOCABULARY = [
    # desk / office
    "sticky note", "notebook", "pen", "pencil", "colored pencil", "marker",
    "highlighter", "eraser", "ruler", "stapler", "paper clip", "binder clip",
    "folder", "envelope", "clipboard", "whiteboard", "calculator", "tape dispenser",
    "rubber band", "push pin", "index card", "notepad", "desk lamp", "pencil case",
    # toys / play
    "lego brick", "rubber duck", "toy car", "toy block", "building block",
    "action figure", "doll", "stuffed animal", "teddy bear", "puzzle piece",
    "playing card", "dice", "yo-yo", "spinning top", "toy train", "marble",
    "crayon", "modeling clay", "balloon", "party hat", "gift box",
    # kitchen / dining
    "mug", "coffee cup", "drinking glass", "water bottle", "thermos", "plate",
    "saucer", "teapot", "kettle", "cutting board", "mixing bowl", "whisk",
    "spatula", "ladle", "measuring cup", "oven mitt", "dish towel", "sponge",
    "dish soap bottle", "jar", "food container", "lunch box", "egg carton",
    "coaster", "napkin", "straw", "chopsticks", "ice cube tray", "salt shaker",
    # bathroom / personal care
    "toothbrush", "toothpaste tube", "soap bar", "shampoo bottle", "towel",
    "hairbrush", "comb", "razor", "cotton swab", "lotion bottle", "perfume bottle",
    "nail polish bottle", "lipstick", "makeup brush", "hand mirror",
    # clothing / accessories
    "t-shirt", "hoodie", "jacket", "sweater", "scarf", "beanie", "cap", "hat",
    "glove", "mitten", "sock", "shoe", "sneaker", "boot", "sandal", "slipper",
    "belt", "wallet", "purse", "tote bag", "shopping bag", "sunglasses",
    "wristwatch", "bracelet", "necklace", "headband", "hair clip", "shoelace",
    # electronics / cables
    "phone case", "charging cable", "power adapter", "extension cord",
    "power strip", "headphones", "earbuds", "computer mouse", "mousepad",
    "usb drive", "memory card", "game controller", "remote control",
    "light bulb", "flashlight", "lantern", "battery", "alarm clock", "speaker",
    "webcam", "microphone", "router", "calculator watch", "e-reader",
    # tools / garage
    "screwdriver", "wrench", "hammer", "pliers", "tape measure", "utility knife",
    "paintbrush", "paint roller", "paint can", "spray can", "bucket", "funnel",
    "duct tape", "rope", "bungee cord", "zip tie", "clamp", "sandpaper",
    "safety helmet", "safety vest", "work glove", "toolbox", "ladder", "broom",
    "dustpan", "mop", "watering can", "garden hose", "flower pot", "trowel",
    # outdoor / street
    "traffic cone", "mailbox", "trash can", "recycling bin", "street sign",
    "flag", "umbrella", "tent", "sleeping bag", "backpack", "cooler",
    "picnic blanket", "frisbee", "beach ball", "kite", "skateboard", "scooter",
    "helmet", "water gun", "pool float", "life jacket", "fishing rod",
    # food (colour-distinct, non-COCO)
    "bell pepper", "tomato", "lemon", "lime", "strawberry", "grape", "cherry",
    "watermelon", "pineapple", "mango", "peach", "plum", "pumpkin", "corn",
    "eggplant", "cucumber", "lettuce", "cabbage", "candy", "lollipop",
    "gummy bear", "chocolate bar", "cookie", "cupcake", "macaron", "jelly",
    "soda can", "juice box", "cereal box", "chip bag", "egg", "bread loaf",
    # home / furniture-adjacent
    "pillow", "cushion", "blanket", "curtain", "rug", "doormat", "picture frame",
    "candle", "vase", "basket", "laundry basket", "clothespin", "hanger",
    "storage box", "cardboard box", "book", "magazine", "board game box",
    "photo album", "wall clock", "door handle", "key", "keychain", "coin",
    # crafts / misc
    "yarn ball", "thread spool", "button", "ribbon", "wrapping paper",
    "origami paper", "sticker", "stamp", "paint palette", "canvas",
    "flower", "tulip", "rose", "sunflower", "leaf", "pinecone", "seashell",
    "rock", "brick", "tile", "sponge ball", "tennis ball", "golf ball",
    "ping pong ball", "shuttlecock", "whistle", "medal", "trophy",
]
