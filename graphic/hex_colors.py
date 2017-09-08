"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
"""


class Color:
    """This class provides one with pre-chosen colors."""

    Red = ["#b71c1c", "#d32f2f", "#f44336", "#e57373", "#ef9a9a"]
    Pink = ["#880E4F", "#C2185B", "#E91E63", "#F06292", "#F48FB1"]
    Purple = ["#4A148C", "#7B1FA2", "#9C27B0", "#BA68C8", "#CE93D8"]
    DeepPink = ["#311B92", "#512DA8", "#673AB7", "#9575CD", "#B39DDB"]
    Indigo = ["#1A237E", "#303F9F", "#3F51B5", "#7986CB", "#9FA8DA"]
    Blue = ["#0D47A1", "#1976D2", "#2196F3", "#64B5F6", "#90CAF9"]
    LightBlue = ["#01579B", "#0288D1", "#03A9F4", "#4FC3F7", "#81D4FA"]
    Cyan = ["#006064", "#0097A7", "#00BCD4", "#4DD0E1", "#80DEEA"]
    Teal = ["#004D40", "#00796B", "#009688", "#4DB6AC", "#80CBC4"]
    Green = ["#1B5E20", "#388E3C", "#4CAF50", "#81C784", "#A5D6A7"]
    LightGreen = ["#33691E", "#689F38", "#8BC34A", "#AED581", "#C5E1A5"]
    Lime = ["#827717", "#AFB42B", "#CDDC39", "#DCE775", "#E6EE9C"]
    Yellow = ["#F57F17", "#FBC02D", "#FFEB3B", "#FFF176", "#FFF59D"]
    Amber = ["#FF6F00", "#FFA000", "#FFC107", "#FFD54F", "#FFE082"]
    Orange = ["#E65100", "#F57C00", "#FF9800", "#FFB74D", "#FFCC80"]
    DeepOrange = ["#BF360C", "#E64A19", "#FF5722", "#FF8A65", "#FFAB91"]
    Brown = ["#3E2723", "#5D4037", "#795548", "#A1887F", "#BCAAA4"]
    BlueGrey = ["#263238", "#455A64", "#607D8B", "#90A4AE", "#B0BEC5"]

    C_36H = ["#000080", "#0000FF", "#1E90FF", "#87CEFA", "#00CED1", "#00FFFF",
             "#228B22", "#32CD32", "#9ACD32", "#66CDAA", "#00FF7F", "#7FFF00",
             "#8B4513", "#D2691E", "#DAA520", "#FFD700", "#FFFF00", "#F0E68C",
             "#FF4500", "#FF8C00", "#B22222", "#CD5C5C", "#FA8072", "#D2B48C",
             "#800080", "#FF1493", "#FF69B4", "#DA70D6", "#FFC0CB", "#DDA0DD",
             "#000000", "#696969", "#A9A9A9", "#A9A9A9", "#D3D3D3", "#C0C0C0"]

    # ==> 36 Colors!
    # "NAVY", "BLUE", "DODGERBLUE", "LIGHTSKYBLUE", "DARKTURQUOISE", "CYAN",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN", "MEDIUMAQUAMARINE", "SPRINGGREEN", "CHARTREUSE",
    # "SADDLEBROWN", "CHOCOLATE", "GOLDENROD", "GOLD", "YELLOW", "KHAKI",
    # "ORANGERED", "DARKORANGE", "FIREBRICK", "INDIANRED", "SALMON", "TAN",
    # "PURPLE", "DEEPPINK", "HOTPINK", "ORCHID", "PINK", "PLUM",
    # "BLACK", "DIMGRAY", "DARKGRAY", "GRAY", "LIGHTGRAY", "SILVER"

    C_35H = ["#000080", "#0000FF", "#1E90FF", "#87CEFA", "#008B8B", "#00CED1", "#00FFFF",
             "#228B22", "#32CD32", "#9ACD32", "#3CB371", "#66CDAA", "#00FF7F", "#7FFF00",
             "#8B4513", "#D2691E", "#DAA520", "#FFD700", "#FFFF00", "#D2B48C", "#F0E68C",
             "#B22222", "#FF4500", "#FF8C00", "#CD5C5C", "#FA8072", "#E9967A", "#FFE4E1",
             "#000000", "#696969", "#A9A9A9", "#A9A9A9", "#D3D3D3", "#D8BFD8", "#C0C0C0"]

    # ==> 35 Colors!
    # "NAVY", "BLUE", "DODGERBLUE", "LIGHTSKYBLUE", "DARKCYAN", "DARKTURQUOISE", "CYAN",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN", "MEDIUMSEAGREEN", "MEDIUMAQUAMARINE", "SPRINGGREEN", "CHARTREUSE",
    # "SADDLEBROWN", "CHOCOLATE", "GOLDENROD", "GOLD", "YELLOW", "TAN", "KHAKI",
    # "FIREBRICK", "ORANGERED", "DARKORANGE", "INDIANRED", "SALMON", "darksalmon", "MISTYROSE",
    # "PURPLE", "DEEPPINK", "HOTPINK", "ORCHID", "PINK", "THISTLE", "PLUM"

    C_30H = ["#000080", "#0000FF", "#1E90FF", "#87CEFA", "#00CED1", "#00FFFF",
             "#228B22", "#32CD32", "#9ACD32", "#66CDAA", "#00FF7F", "#7FFF00",
             "#8B4513", "#D2691E", "#DAA520", "#FFD700", "#FFFF00", "#F0E68C",
             "#FF4500", "#FF8C00", "#B22222", "#CD5C5C", "#FA8072", "#D2B48C",
             "#800080", "#FF1493", "#FF69B4", "#DA70D6", "#FFC0CB", "#DDA0DD"]

    # ==> 30 Colors!
    # "NAVY", "BLUE", "DODGERBLUE", "LIGHTSKYBLUE", "DARKTURQUOISE", "CYAN",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN", "MEDIUMAQUAMARINE", "SPRINGGREEN", "CHARTREUSE",
    # "SADDLEBROWN", "CHOCOLATE", "GOLDENROD", "GOLD", "YELLOW", "KHAKI",
    # "ORANGERED", "DARKORANGE", "FIREBRICK", "INDIANRED", "SALMON", "TAN",
    # "PURPLE", "DEEPPINK", "HOTPINK", "ORCHID", "PINK", "PLUM"

    C_24H = ["#000080", "#0000FF", "#1E90FF", "#87CEFA", "#00CED1", "#00FFFF",
             "#228B22", "#32CD32", "#9ACD32", "#66CDAA", "#00FF7F", "#7FFF00",
             "#FF4500", "#FF8C00", "#B22222", "#CD5C5C", "#FA8072", "#D2B48C",
             "#800080", "#FF1493", "#FF69B4", "#DA70D6", "#FFC0CB", "#DDA0DD"]

    # ==> 24 Colors!
    # "NAVY", "BLUE", "DODGERBLUE", "LIGHTSKYBLUE", "DARKTURQUOISE", "CYAN",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN", "MEDIUMAQUAMARINE", "SPRINGGREEN", "CHARTREUSE",
    # "ORANGERED", "DARKORANGE", "FIREBRICK", "INDIANRED", "SALMON", "TAN",
    # "PURPLE", "DEEPPINK", "HOTPINK", "ORCHID", "PINK", "PLUM"

    C_18H = ["#000080", "#0000FF", "#1E90FF", "#87CEFA", "#00CED1", "#00FFFF",
             "#228B22", "#32CD32", "#9ACD32", "#66CDAA", "#00FF7F", "#7FFF00",
             "#FF4500", "#FF8C00", "#B22222", "#CD5C5C", "#FA8072", "#D2B48C"]

    # ==> 18 Colors!
    # "NAVY", "BLUE", "DODGERBLUE", "LIGHTSKYBLUE", "DARKTURQUOISE", "CYAN",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN", "MEDIUMAQUAMARINE", "SPRINGGREEN", "CHARTREUSE",
    # "ORANGERED", "DARKORANGE", "FIREBRICK", "INDIANRED", "SALMON", "TAN",

    C_12H = ["#000080", "#0000FF", "#1E90FF",
             "#228B22", "#32CD32", "#9ACD32",
             "#FF4500", "#FF8C00", "#B22222",
             "#800080", "#FF1493", "#FF69B4"]

    # ==> 12 Colors!
    # "NAVY", "BLUE", "DODGERBLUE",
    # "FORESTGREEN", "LIMEGREEN", "YELLOWGREEN",
    # "ORANGERED", "DARKORANGE", "FIREBRICK",
    # "PURPLE", "DEEPPINK", "HOTPINK"

    C_6H = ["#000080", "#1E90FF",
            "#228B22", "#FFD700",
            "#FF1493", "#FF4500"]

    # ==> 6 Colors!
    # "NAVY", "DODGERBLUE",
    # "FORESTGREEN", "GOLD",
    # "DEEPPINK", "ORANGERED",

    C_5H = ["#000080", "#1E90FF", "#32CD32", "#FF1493", "#B22222"]

    # ==> 5 Colors!
    # "FIREBRICK", "DEEPPINK", "LIMEGREEN", "DODGERBLUE", "NAVY"

    C_3H = ["#0000FF", "#FF0000", "#FFD700"]
    # ==> 3 Colors!
    # "BLUE", "RED", "GOLD"

    # ==> 2 Colors!
    C_2H = ["#0000FF", "#FF0000"]