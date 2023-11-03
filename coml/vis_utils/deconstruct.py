# deconstruct svg to spec
import math
import re
from datetime import datetime
from xml.dom import minidom

import numpy as np
from dateutil.parser import parse

keys_of_interest = [
    "id",
    "style",
    "transform",
    "translate",
    "rotate",
    "skewX",
    "skewY",
    "x",
    "dx",
    "dy",
    "y",
    "cx",
    "cy",
    "r",
    "x1",
    "x2",
    "y1",
    "y2",
    "width",
    "height",
    "fill",
    "color",
    "opacity",
    "fill-opacity",
    "stroke",
    "stroke-width",
    "stroke-opacity",
    "font-size",
    "font-weight",
    "text-anchor",
    "aria-label",
    "display",
    "matrix",
]
marks = ["bar", "arc", "circle", "line", "text"]
channels = ["fill", "width", "height", "r", "stroke", "stroke-width"]
PRECISION = 0.0001


def extract_tag(node):
    return node.tagName


def get_attribute_names(node):
    return node.attributes.keys()


def get_attribute_value(node, attr):
    return node.getAttribute(attr)


def process_text_matplotlib(node, spec, acc_trans):
    for child in node.childNodes:
        # Pass Text node \n
        is_first_line = True
        if child.nodeType != child.TEXT_NODE:
            # the fist node is comment node
            if child.nodeType == child.COMMENT_NODE:
                if "content" in spec:
                    spec["content"] += "\n" + child.data.strip()
                else:
                    spec["content"] = child.data.strip()
                    spec["tag"] = "text"
            if child.nodeType == child.ELEMENT_NODE and is_first_line:
                if extract_tag(child) == "g":
                    # process transform
                    process_transform(child, spec, acc_trans)
                    if "x" not in spec:
                        spec["x"] = 0
                    else:
                        spec["x"] = float(spec["x"])
                    if "y" not in spec:
                        spec["y"] = 0
                    else:
                        spec["y"] = float(spec["y"])

                    spec["x"] += spec["translate"][0]
                    spec["y"] += spec["translate"][1]
                    del spec["translate"]
                    is_first_line = False


def process_path(node, spec):
    allowed_operator = [
        "M",
        "m",
        "L",
        "l",
        "V",
        "v",
        "H",
        "h",
        "Z",
        "z",
        "A",
        "a",
        "C",
        "c",
    ]
    draw_seq = get_attribute_value(node, "d")
    point_seq = []
    char_seq = ""
    notCircle = (
        "L" in draw_seq
        or "l" in draw_seq
        or "V" in draw_seq
        or "v" in draw_seq
        or "H" in draw_seq
        or "h" in draw_seq
    )
    for i in range(len(draw_seq)):
        char = draw_seq[i]
        if char in allowed_operator:
            char_seq += char
            # close path
            if char == "Z" or char == "z":
                if len(point_seq) == 0:
                    break
                if (
                    point_seq[0][0] != point_seq[-1][0]
                    or point_seq[0][1] != point_seq[-1][1]
                ):
                    point_seq.append(point_seq[0])
            else:
                j = i + 1
                # drawSeq[j] is not letter
                while re.match(r"[^a-zA-Z]", draw_seq[j]):
                    j += 1
                    if j >= len(draw_seq):
                        break
                shifts = draw_seq[i + 1 : j].strip()
                if "," in shifts:
                    separator = shifts.index(",")
                else:
                    separator = -1
                if separator != -1:
                    af = list(filter(lambda d: d != "", shifts.split(",")))
                else:
                    af = list(filter(lambda d: d != "", shifts.split(" ")))
                if len(point_seq) > 0:
                    previous_x = point_seq[-1][0]
                    previous_y = point_seq[-1][1]

                if char == "A" and not notCircle:  # arc
                    if (
                        separator > 0 and shifts.index(" ") > 0
                    ):  # some cases have "a 0,0 0 0 1 0,0"
                        continue
                    point_seq.append([float(af[-2]), float(af[-1])])
                elif char == "a" and not notCircle:
                    if (
                        separator > 0 and shifts.index(" ") > 0
                    ):  # some cases have "a 0,0 0 0 1 0,0"
                        continue
                    point_seq.append(
                        [previous_x + float(af[-2]), previous_y + float(af[-1])]
                    )
                elif char == "M" or char == "L":  # move to or line to
                    point_seq.append([float(af[0]), float(af[1])])
                    af = af[2:]
                    if char == "L" and len(af) % 2 == 0 and len(af) > 0:
                        while len(af) > 0:
                            point_seq.append([float(af[0]), float(af[1])])
                            af = af[2:]
                elif char == "m" or char == "l":
                    point_seq.append(
                        [previous_x + float(af[0]), previous_y + float(af[1])]
                    )
                elif char == "C":  # cubic-bezier
                    point_seq.append([float(af[-2]), float(af[-1])])
                elif char == "c":
                    point_seq.append(
                        [previous_x + float(af[-2]), float(previous_y + af[-1])]
                    )
                elif char == "V":  # vertical line
                    point_seq.append([previous_x, float(shifts)])
                elif char == "v":
                    # if (parseFloat(shifts) !== 0)
                    # tbd: here we cannot use this if otherwise 0hw-rect will be filtered out;
                    point_seq.append([previous_x, previous_y + float(shifts)])
                elif char == "H":  # horizontal line
                    point_seq.append([float(shifts), previous_y])
                elif char == "h":
                    point_seq.append([previous_x + float(shifts), previous_y])

    # then check whether the point seq forms a rect or a line
    # haven't deal with all possible cases
    if (
        len(point_seq) == 5
        and point_seq[0][0] == point_seq[-1][0]
        and point_seq[0][1] == point_seq[-1][1]
    ) and (char_seq[0:4] == "MLLL" or char_seq[0:4] == "Mlll"):
        width = (
            max(point_seq, key=lambda p: p[0])[0]
            - min(point_seq, key=lambda p: p[0])[0]
        )
        height = (
            max(point_seq, key=lambda p: p[1])[1]
            - min(point_seq, key=lambda p: p[1])[1]
        )
        spec["tag"] = "rect"
        spec["x"] = min(point_seq, key=lambda p: p[0])[0]
        spec["y"] = min(point_seq, key=lambda p: p[1])[1]
        spec["width"] = width
        spec["height"] = height
    elif len(point_seq) == 2:
        spec["tag"] = "line"
        spec["x1"] = point_seq[0][0]
        spec["y1"] = point_seq[0][1]
        spec["x2"] = point_seq[1][0]
        spec["y2"] = point_seq[1][1]
    elif len(point_seq) == 3 and (char_seq == "MAA" or char_seq == "Maa"):
        if point_seq[0][0] == point_seq[2][0] and point_seq[0][1] == point_seq[2][1]:
            spec["tag"] = "circle"
            spec["cx"] = 0.5 * (point_seq[0][0] + point_seq[1][0])
            spec["cy"] = 0.5 * (point_seq[0][1] + point_seq[1][1])
            # todo: radius
            spec["r"] = 0.5 * (point_seq[0][0] - point_seq[1][0])
    elif re.match(r"^M(C|c){8}(Z|z)$", char_seq):
        if abs(point_seq[0][0] - point_seq[2][0]) == abs(
            point_seq[0][1] - point_seq[2][1]
        ):
            spec["tag"] = "circle"
            spec["r"] = abs(point_seq[0][0] - point_seq[2][0])
            spec["cx"] = 0.5 * (point_seq[0][0] + point_seq[4][0])
            spec["cy"] = 0.5 * (point_seq[0][1] + point_seq[4][1])
    elif re.match(r"^M(C|c)+(L|l)z$", char_seq):
        if len(point_seq) % 2 == 1:
            mid = int((len(point_seq) - 1) / 2) - 1
            # 90 degree
            if (
                abs(
                    (point_seq[0][0] - point_seq[-3][0])
                    * (point_seq[mid][0] - point_seq[-2][0])
                    + (point_seq[0][1] - point_seq[-3][1])
                    * (point_seq[mid][1] - point_seq[-2][1])
                )
                < 0.001
            ):
                spec["tag"] = "arc"
                spec["cx"] = point_seq[-2][0]
                spec["cy"] = point_seq[-2][1]
                spec["arc"] = [point_seq[0], point_seq[mid], point_seq[-3]]
                spec["r"] = (
                    (point_seq[0][0] - point_seq[-2][0]) ** 2
                    + (point_seq[0][1] - point_seq[-2][1]) ** 2
                ) ** 0.5
                spec["theta"] = math.acos(
                    (
                        (point_seq[0][0] - point_seq[-2][0])
                        * (point_seq[-3][0] - point_seq[-2][0])
                        + (point_seq[0][1] - point_seq[-2][1])
                        * (point_seq[-3][1] - point_seq[-2][1])
                    )
                    / (spec["r"] ** 2)
                )
    # below is for handling cases where using a rect-like shapes to represent axis lines
    elif len(point_seq) > 1:
        width = (
            max(point_seq, key=lambda p: p[0])[0]
            - min(point_seq, key=lambda p: p[0])[0]
        )
        height = (
            max(point_seq, key=lambda p: p[1])[1]
            - min(point_seq, key=lambda p: p[1])[1]
        )
        # small width or height and small gap between points
        if width < 2 and np.max(np.diff(np.array([p[1] for p in point_seq]))) < 20:
            spec["tag"] = "line"
            spec["x1"] = max(point_seq, key=lambda p: p[0])[0]
            spec["y1"] = min(point_seq, key=lambda p: p[1])[1]
            spec["x2"] = max(point_seq, key=lambda p: p[0])[0]
            spec["y2"] = max(point_seq, key=lambda p: p[1])[1]
        elif height < 2 and np.max(np.diff(np.array([p[0] for p in point_seq]))) < 20:
            spec["tag"] = "line"
            spec["x1"] = min(point_seq, key=lambda p: p[0])[0]
            spec["y1"] = min(point_seq, key=lambda p: p[1])[1]
            spec["x2"] = max(point_seq, key=lambda p: p[0])[0]
            spec["y2"] = min(point_seq, key=lambda p: p[1])[1]
        elif re.match(r"^(M|m|L|l)+", char_seq):
            spec["tag"] = "path"
            spec["points"] = point_seq
    if spec["tag"] == "path":
        spec["d"] = draw_seq
    return spec


def process_axis_matplotlib(spec):
    if "children" not in spec or len(spec["children"]) == 0:
        return
    spec["type"] = "axis"
    spec["tick"] = []
    spec["tick_value"] = []
    for child in spec["children"]:
        child_id = child["id"]
        if "xtick" in child_id:
            spec["type"] = "xaxis"
        elif "ytick" in child_id:
            spec["type"] = "yaxis"

        if "tick" in child_id:
            for child2 in child["children"]:
                if child2["tag"] == "line":
                    if spec["type"] == "xaxis":
                        spec["tick_value"].append(child2["x1"])
                    elif spec["type"] == "yaxis":
                        spec["tick_value"].append(child2["y1"])
                if child2["tag"] == "text":
                    spec["tick"].append(child2["content"])

        elif "text" in child_id:
            if re.match(r"^1e[1-9][0-9]?$", child["content"]):
                spec["unit"] = eval(child["content"])
            else:
                spec["title"] = child["content"]


def process_legend_matplotlib(spec):
    spec["type"] = "legend"
    labels = []
    examples = []
    # element(index 0) might be background
    # todo: A more accurate way to recognize background
    first = 0 if spec["children"][0]["tag"] == "text" else 1
    for i in range(first, len(spec["children"])):
        child = spec["children"][i]
        if child["tag"] == "text":
            labels.append(child)
        else:
            examples.append(child)

    # if example is group, combination
    if len(examples) > 0 and examples[0]["tag"] == "g":
        # hypothesis: line > circle
        children_tags = [child["tag"] for child in examples[0]["children"]]
        if "line" in children_tags:
            child_tag = "line"
        new_examples = []
        for example in examples:
            for child in example["children"]:
                if child["tag"] == child_tag:
                    new_examples.append(child)
                    break
        if len(new_examples) == len(examples):
            examples = new_examples

    differences = []
    for channel in channels:
        if channel in examples[0]:
            value = examples[0][channel]
            is_str = isinstance(value, str)
            for i in range(1, len(examples)):
                if (is_str and examples[i][channel] != value) or (
                    not is_str and abs(examples[i][channel] - value) > PRECISION
                ):
                    differences.append(channel)
                    break

    # # hypothesis: legend label in same row or column
    # todo: bug: 1 title + 1 label
    if len(labels) > len(examples) and len(labels) > 2:
        labels_x = [label["x"] for label in labels]
        labels_y = [label["y"] for label in labels]
        for i in range(len(labels) - 1, -1, -1):
            if labels_x.count(labels_x[i]) == 1 and labels_y.count(labels_y[i]) == 1:
                # maybe title
                spec["title"] = labels[i]["content"]
                del labels[i]

    dists = []
    for i in range(len(examples)):
        if examples[i]["tag"] == "rect":
            dists = [
                (label["x"] - examples[i]["x"]) ** 2
                + (label["y"] - examples[i]["y"]) ** 2
                for label in labels
            ]
        elif examples[i]["tag"] == "line":
            dists = [
                (label["x"] - examples[i]["x1"]) ** 2
                + (label["y"] - examples[i]["y1"]) ** 2
                for label in labels
            ]
        elif examples[i]["tag"] == "circle":
            dists = [
                (label["x"] - examples[i]["cx"]) ** 2
                + (label["y"] - examples[i]["cy"]) ** 2
                for label in labels
            ]

        if len(dists) > 0:
            min_index = dists.index(min(dists))
            examples[i]["label"] = labels[min_index]["content"]

    mapping = {}
    # remove redundancy
    if "fill" in differences and "stroke" in differences:
        flag = True
        for example in examples:
            if example["stroke"] != example["fill"]:
                flag = False
                break
        if flag:
            differences.remove("stroke")

    for example in examples:
        key = ""
        for difference in differences:
            key += difference + ":" + str(example[difference]) + ";"
        mapping[key] = example["label"]
    spec["mapping"] = mapping
    spec["channel"] = differences


def process_transform(node, spec, acc_trans):
    trans_str = get_attribute_value(node, "transform")
    if "translate" in trans_str:
        index_of_translate = trans_str.index("translate")
        new_str = trans_str[index_of_translate + 9 :]
        index_of_right_bracket = new_str.index(")")
        index_of_left_bracket = new_str.index("(")
        value = new_str[index_of_left_bracket + 1 : index_of_right_bracket].strip()
        if "," in value:
            dxy = value.split(",")
            dx = float(dxy[0])
            dy = float(dxy[1])
        else:
            dxy = value.split(" ")
            dx = float(dxy[0])
            if len(dxy) == 1:
                dy = 0.0
            else:
                dy = float(dxy[-1])
        spec["translate"] = [dx + acc_trans[0], dy + acc_trans[1]]
    if "matrix" in trans_str:
        index_of_matrix = trans_str.index("matrix")
        new_str = trans_str[index_of_matrix + 6 :]
        index_of_right_bracket = new_str.index(")")
        index_of_left_bracket = new_str.index("(")
        value = new_str[index_of_left_bracket + 1 : index_of_right_bracket].strip()
        if "," in value:
            matrix = value.split(",")
        else:
            matrix = value.split(" ")
        if "matrix" not in spec:
            spec["matrix"] = matrix
        else:
            # todo: concatenate matrix
            spec["matrix"] = matrix + spec["matrix"]
    # if 'rotate' in trans_str:
    #     spec['rotate'] = True
    # if 'scale' in trans_str:
    #     spec['scale'] = True
    # if 'skewX' in trans_str:
    #     spec['skewX'] = True
    # if 'skewY' in trans_str:
    #     spec['skewY'] = True


def extract_features(node, spec, parent, acc_trans, global_style):
    # analysis attributes
    attrs = get_attribute_names(node)
    if spec["tag"] == "style":
        pass
        # global_style = style_parser(node.textContent)
    # todo: class or tag based style

    # process path
    if spec["tag"] == "path":
        process_path(node, spec)

    # todo: process polyline

    # inheriting parent's attributes
    if parent and parent["tag"] == "g":
        for key in parent:
            if (
                key in keys_of_interest
                and key != "id"
                and key != "width"
                and key != "height"
            ):
                spec[key] = parent[key]

    for attr in keys_of_interest:
        if attr in attrs:
            if attr == "style":
                local_styles = get_attribute_value(node, attr).split(";")
                for pair in local_styles:
                    if pair == "":
                        continue
                    pair = pair.strip().split(":")
                    if len(pair) == 2:
                        if pair[0].strip() in keys_of_interest:
                            spec[pair[0].strip()] = pair[1].strip()
            elif attr == "transform":
                # process transform
                process_transform(node, spec, acc_trans)
            elif attr == "font-size":
                pass
            elif attr == "dx" or attr == "dy":
                pass
            elif attr == "x" or attr == "y":
                spec[attr] = float(get_attribute_value(node, attr))
            else:
                spec[attr] = get_attribute_value(node, attr).strip()

    if "fill-opacity" in spec:
        if spec["fill-opacity"] == "0" and "fill" in parent:
            spec["fill"] = parent["fill"]
            del spec["fill-opacity"]

    if (
        ("transform" not in attr or get_attribute_value(node, "transform") == "")
        and acc_trans[0] != 0
        and acc_trans[1] != 0
    ):
        spec["transform"] = f"translate({acc_trans[0]}, {acc_trans[1]})"


# node: element node
# acc_trans: accumulated transformation
# global_style: global style
def parser_node(
    node, parent, defss={}, acc_trans=[0, 0], global_style={}, source="matplotlib"
):
    spec = {}
    spec["children"] = []

    tag = extract_tag(node)
    spec["tag"] = tag

    if tag == "metadata":
        return spec

    if tag == "style":
        # todo: global style
        pass

    extract_features(node, spec, parent, acc_trans, global_style)

    if tag == "defs":
        # todo: defs
        for child in node.childNodes:
            if child.nodeType == child.ELEMENT_NODE:
                child_spec = parser_node(child, spec, defss)
                if "id" in child_spec:
                    defss[child_spec["id"]] = child_spec
        return spec

    if tag == "use":
        defs = defss[get_attribute_value(node, "xlink:href")[1:]]
        if defs:  # copy
            spec["tag"] = defs["tag"]
            for key in defs:
                if key != "id" and key not in spec:
                    spec[key] = defs[key]
        # todo: transform
        if "x" in spec:
            if spec["tag"] == "line":
                spec["x1"] += spec["x"]
                spec["x2"] += spec["x"]
                del spec["x"]
            elif spec["tag"] == "circle":
                spec["cx"] += spec["x"]
                del spec["x"]
        if "y" in spec:
            if spec["tag"] == "line":
                spec["y1"] += spec["y"]
                spec["y2"] += spec["y"]
                del spec["y"]
            elif spec["tag"] == "circle":
                spec["cy"] += spec["y"]
                del spec["y"]

    # process text
    if tag == "g" and source == "matplotlib":
        node_id = node.getAttribute("id")
        if "text" in node_id:
            process_text_matplotlib(node, spec, acc_trans)

    if node.childNodes.length > 0 and spec["tag"] != "text":
        for child in node.childNodes:
            # Pass Text node \n
            if child.nodeType == child.ELEMENT_NODE:
                parser_node(child, spec, defss, acc_trans, global_style, source)
        # remove useless children
        if len(spec["children"]) == 1 and spec["tag"] == "g":
            if "id" not in spec["children"][0]:
                if "id" in spec:
                    spec["children"][0]["id"] = spec["id"]
                spec = spec["children"][0]

    # process matplotlib
    if spec["tag"] == "g" and source == "matplotlib" and "type" not in spec:
        node_id = node.getAttribute("id")
        # remove background
        if "figure" in node_id:
            spec["type"] = "figure"
            if spec["children"][0]["tag"] == "rect":
                spec["width"] = spec["children"][0]["width"]
                spec["height"] = spec["children"][0]["height"]
                del spec["children"][0]
        elif "axes" in node_id:
            spec["type"] = "subplot"
            # remove background
            if spec["children"][0]["tag"] == "rect":
                spec["width"] = spec["children"][0]["width"]
                spec["height"] = spec["children"][0]["height"]
                spec["x"] = spec["children"][0]["x"]
                spec["y"] = spec["children"][0]["y"]
                del spec["children"][0]
                # remove axis lines
                for index in range(len(spec["children"]) - 2, -1, -1):
                    child = spec["children"][index]
                    if child["tag"] == "line":
                        if (
                            abs(child["x1"] - child["x2"]) == spec["width"]
                            and child["y1"] == child["y2"]
                        ):
                            del spec["children"][index]
                        elif (
                            abs(child["y1"] - child["y2"]) == spec["height"]
                            and child["x1"] == child["x2"]
                        ):
                            del spec["children"][index]
        elif "matplotlib.axis" in node_id:
            process_axis_matplotlib(spec)
        elif "legend" in node_id:
            process_legend_matplotlib(spec)

    if "children" in spec and len(spec["children"]) == 0:
        del spec["children"]

    if parent is not None:
        if spec["tag"] == "g" and (
            "children" not in spec or len(spec["children"]) == 0
        ):
            pass
        else:
            parent["children"].append(spec)
    return spec


def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_datetime(s):
    # treat month as nominal
    mons = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    if s[0:3].capitalize() in mons:
        return False
    try:
        parse(s)
        return True
    except ValueError:
        return False


def check_time_format(time_str, format):
    try:
        datetime.strptime(time_str, format)
        return True
    except ValueError:
        return False


def analysis_data_type(data_domain, data_range):
    # range not number -> nominal
    numeric_values = [is_numeric(s) for s in data_range]
    if False in numeric_values:
        return "nominal"

    if len(data_domain) == len(set(data_domain)):
        numeric_values = [is_numeric(s) for s in data_domain]
        if False not in numeric_values:
            return "quantitative"

        datetime_values = [is_datetime(s) for s in data_domain]
        if False not in datetime_values:
            return "temporal"
    return "nominal"


def analysis_axis(spec, encoding):
    if spec["type"] == "xaxis":
        channel = "x"
    elif spec["type"] == "yaxis":
        channel = "y"

    if channel:
        encoding[channel] = {
            "field": "field_" + channel,
            "type": analysis_data_type(spec["tick"], spec["tick_value"]),
            "scale": {
                "domain": spec["tick"],
                "range": spec["tick_value"],
            },
        }
        if "unit" in spec:
            encoding[channel]["unit"] = spec["unit"]
        if "title" in spec:
            encoding[channel]["title"] = spec["title"]


def analysis_legend(spec, encoding):
    if len(spec["channel"]) == 1:
        channel = spec["channel"][0]
        data_domain = []
        data_range = []
        for key in spec["mapping"]:
            data_domain.append(spec["mapping"][key])
            data_range.append(key.split(";")[0].split(":")[1])
        encoding[channel] = {
            "field": "field_" + channel,
            "scale": {
                "domain": data_domain,
                "range": data_range,
            },
            "type": analysis_data_type(data_domain, data_range),
        }
        if "title" in spec:
            encoding[channel]["title"] = spec["title"]


def get_aligned_index(rect, channel, ticks):
    if channel == "x":
        mid = rect["x"] + 0.5 * rect["width"]
    elif channel == "y":
        mid = rect["y"] + 0.5 * rect["height"]

    if mid is not None:
        if len(ticks) == 1:
            if abs(mid - ticks[0]) < PRECISION:
                return 0
        else:
            tick_width = (max(ticks) - min(ticks)) / (len(ticks) - 1)
            minus = np.array([abs(mid - tick) for tick in ticks])
            indexes = np.where(minus < (PRECISION + tick_width / 2))

            if len(indexes[0]) > 0:
                return indexes[0][0]
    return None


def is_align_with_axis(rect, channel, ticks):
    # todo: handle ticks without tick line
    if len(ticks) == 0:
        return True
    if get_aligned_index(rect, channel, ticks) is not None:
        return True
    return False


def identify_mark_circles(circles, spec):
    encoding = spec["encoding"]
    filtered_circles = circles

    if "x" in encoding and "y" in encoding:
        if (
            encoding["x"]["type"] == "quantitative"
            and encoding["y"]["type"] == "quantitative"
        ):
            # in range
            x_range = encoding["x"]["scale"]["range"]
            y_range = encoding["y"]["scale"]["range"]
            filtered_circles = list(
                filter(
                    lambda circle: circle["cx"] >= x_range[0]
                    and circle["cx"] <= x_range[1]
                    and circle["cy"] >= y_range[1]
                    and circle["cy"] <= y_range[0],
                    filtered_circles,
                )
            )

        for channel in encoding:
            if channel != "x" and channel != "y":
                if encoding[channel]["type"] == "nominal":
                    filtered_circles = list(
                        filter(
                            lambda circle: circle[channel]
                            in encoding[channel]["scale"]["range"],
                            filtered_circles,
                        )
                    )

    return filtered_circles


def identify_mark_bars(rects, spec):
    encoding = spec["encoding"]

    if "x" in encoding and "y" in encoding:
        x_range = encoding["x"]["scale"]["range"]
        y_range = encoding["y"]["scale"]["range"]

        # ## vertical bar?
        # y: in range
        filtered_rects = rects
        filtered_rects = list(
            filter(
                lambda rect: rect["y"] >= y_range[1]
                and rect["y"] + rect["height"] <= y_range[0],
                filtered_rects,
            )
        )
        if len(filtered_rects) > 0:
            width = [rect["width"] for rect in filtered_rects]
            # same width ?
            if max(width) - min(width) > PRECISION:
                filtered_rects = []

        if len(filtered_rects) == 0:
            # ## horizontal bar?
            # x: in range
            filtered_rects = rects
            filtered_rects = list(
                filter(
                    lambda rect: rect["x"] >= x_range[0]
                    and rect["x"] + rect["width"] <= x_range[1],
                    filtered_rects,
                )
            )
            if len(filtered_rects) > 0:
                height = [rect["height"] for rect in filtered_rects]
                # same height ?
                if max(height) - min(height) > PRECISION:
                    return []

    for channel in encoding:
        if channel != "x" and channel != "y":
            if encoding[channel]["type"] == "nominal":
                filtered_rects = list(
                    filter(
                        lambda rect: rect[channel]
                        in encoding[channel]["scale"]["range"],
                        filtered_rects,
                    )
                )

    return filtered_rects


def identify_mark_lines(paths):
    filtered_paths = paths
    filtered_paths = list(filter(lambda path: "points" in path, filtered_paths))
    # points from left to right
    filtered_paths = list(
        filter(
            lambda path: all(
                [
                    path["points"][i][0] < path["points"][i + 1][0]
                    for i in range(len(path["points"]) - 1)
                ]
            ),
            filtered_paths,
        )
    )
    return filtered_paths


def analysis_mark(nodes, spec):
    encoding = spec["encoding"]
    if "arc" in nodes:
        arcs = nodes["arc"]
        # todo: identify arc
        if len(arcs) > 0:
            # add encoding theta
            encoding["theta"] = {
                "field": "field_theta",
                "type": "quantitative",
                "scale": {"domain": [0, 100], "range": [0, 2 * math.pi]},  # percent
            }
            if "fill" not in encoding:
                data_domain = []
                data_range = []
                # find label
                # if no label -> error
                if len(nodes["text"]) >= len(arcs):
                    for arc in arcs:
                        data_range.append(arc["fill"])
                        # value or label?
                        dists = []
                        # filter out value
                        nodes["text"] = [
                            text for text in nodes["text"] if "%" not in text["content"]
                        ]
                        for text in nodes["text"]:
                            # label or title
                            dists.append((arc["arc"][1][1] - text["y"]) ** 2)
                        min_index = dists.index(min(dists))
                        data_domain.append(nodes["text"][min_index]["content"])
                        nodes["text"].pop(min_index)

                encoding["fill"] = {
                    "field": "field_fill",
                    "type": "nominal",
                    "scale": {"domain": data_domain, "range": data_range},
                }
            spec["mark"] = "arc"
            spec["chart"] = "pie"
            spec["data"] = []
            for arc in arcs:
                arc["type"] = "mark"
                item = {}
                for channel in encoding:
                    data_range = encoding[channel]["scale"]["range"]
                    data_domain = encoding[channel]["scale"]["domain"]
                    if encoding[channel]["type"] == "nominal":
                        index = data_range.index(arc[channel])
                        item[encoding[channel]["field"]] = data_domain[index]
                    elif encoding[channel]["type"] == "quantitative":
                        actual_channel = channel
                        item[encoding[channel]["field"]] = round(
                            data_domain[0]
                            + (data_domain[1] - data_domain[0])
                            / (data_range[1] - data_range[0])
                            * arc[actual_channel],
                            6,
                        )

                spec["data"].append(item)

    if ("rect" in nodes) and ("mark" not in spec):
        rects = nodes["rect"]
        # bar chart
        bars = identify_mark_bars(rects, spec)
        if len(bars) > 0:
            spec["mark"] = "bar"
            if len(encoding.keys()) > 2:
                x_mins = [bar["x"] for bar in bars]
                y_mins = [bar["y"] for bar in bars]
                x_maxs = [bar["x"] + bar["width"] for bar in bars]
                y_maxs = [bar["y"] + bar["height"] for bar in bars]
                if (
                    max(x_mins) - min(x_mins) < PRECISION
                    or max(x_maxs) - min(x_maxs) < PRECISION
                    or max(y_mins) - min(y_mins) < PRECISION
                    or max(y_maxs) - min(y_maxs) < PRECISION
                ):
                    spec["chart"] = "grouping bar"
                else:
                    spec["chart"] = "stacked bar"
            else:
                spec["chart"] = "bar"

            width = [bar["width"] for bar in bars]
            if max(width) - min(width) < PRECISION:
                spec["chart"] = f'vertical {spec["chart"]}'
            else:
                spec["chart"] = f'horizontal {spec["chart"]}'
            spec["data"] = []
            for bar in bars:
                bar["type"] = "mark"
                item = {}
                for channel in encoding:
                    data_range = encoding[channel]["scale"]["range"]
                    data_domain = encoding[channel]["scale"]["domain"]
                    if encoding[channel]["type"] == "nominal":
                        if channel == "x" or channel == "y":
                            # todo: Get index in order
                            if len(data_range) == 0 and len(bars) == len(data_domain):
                                index = bars.index(bar)
                            else:
                                index = get_aligned_index(bar, channel, data_range)
                            item[encoding[channel]["field"]] = data_domain[index]
                        else:
                            index = data_range.index(bar[channel])
                            item[encoding[channel]["field"]] = data_domain[index]
                    elif encoding[channel]["type"] == "quantitative":
                        actual_channel = channel
                        invert = 1
                        if channel == "y":
                            invert = -1  # zero point in left top
                            actual_channel = "height"
                        elif channel == "x":
                            actual_channel = "width"

                        if data_range[1] - data_range[0] != 0:
                            item[encoding[channel]["field"]] = round(
                                data_domain[0]
                                + invert
                                * (data_domain[1] - data_domain[0])
                                / (data_range[1] - data_range[0])
                                * (bar[actual_channel]),
                                6,
                            )
                        else:
                            item[encoding[channel]["field"]] = data_range[0]
                spec["data"].append(item)

    if ("circle" in nodes) and ("mark" not in spec):
        circles = nodes["circle"]
        # scatter plot
        circles = identify_mark_circles(circles, spec)
        if len(circles) > 0:
            spec["mark"] = "circle"
            if len(encoding.keys()) > 2:
                spec["chart"] = "grouping scatter"
            else:
                spec["chart"] = "scatter"
            spec["data"] = []
            for circle in circles:
                circle["type"] = "mark"
                item = {}
                for channel in encoding:
                    data_range = encoding[channel]["scale"]["range"]
                    data_domain = encoding[channel]["scale"]["domain"]
                    if encoding[channel]["type"] == "nominal":
                        index = data_range.index(circle[channel])
                        item[encoding[channel]["field"]] = data_domain[index]
                    elif encoding[channel]["type"] == "quantitative":
                        actual_channel = channel
                        if channel == "y":
                            actual_channel = "cy"
                        elif channel == "x":
                            actual_channel = "cx"
                        item[encoding[channel]["field"]] = round(
                            data_domain[0]
                            + (data_domain[1] - data_domain[0])
                            / (data_range[1] - data_range[0])
                            * (circle[actual_channel] - data_range[0]),
                            6,
                        )
                spec["data"].append(item)

    if ("path" or "line" in nodes) and ("mark" not in spec):
        lines = []
        if "path" in nodes:
            paths = nodes["path"]
            lines += identify_mark_lines(paths)
        if "line" in nodes:
            lines += nodes["line"]
            # line chart that only has two points
            if len(encoding["x"]["scale"]["domain"]) == 2:
                for line in nodes["line"]:
                    line["points"] = [
                        [line["x1"], line["y1"]],
                        [line["x2"], line["y2"]],
                    ]

        if len(lines) > 0:
            spec["mark"] = "line"
            if len(encoding.keys()) > 2:
                spec["chart"] = "grouping line"
            else:
                spec["chart"] = "line"
            spec["data"] = []
            for line in lines:
                line["type"] = "mark"
                item = {}
                for channel in encoding:
                    if channel != "x" and channel != "y":
                        data_range = encoding[channel]["scale"]["range"]
                        data_domain = encoding[channel]["scale"]["domain"]
                        if encoding[channel]["type"] == "nominal":
                            index = data_range.index(line[channel])
                            item[encoding[channel]["field"]] = data_domain[index]
                        elif encoding[channel]["type"] == "quantitative":
                            actual_channel = channel
                            item[encoding[channel]["field"]] = round(
                                data_domain[0]
                                + (data_domain[1] - data_domain[0])
                                / (data_range[1] - data_range[0])
                                * (line[actual_channel] - data_range[0]),
                                6,
                            )
                for point in line["points"]:
                    item1 = item.copy()
                    for channel in ["x", "y"]:
                        data_range = encoding[channel]["scale"]["range"]
                        data_domain = encoding[channel]["scale"]["domain"]
                        if channel == "x":
                            value = point[0]
                        else:
                            value = point[1]
                        if encoding[channel]["type"] == "nominal":
                            index = data_range.index(value)
                            item1[encoding[channel]["field"]] = data_domain[index]
                        elif encoding[channel]["type"] == "quantitative":
                            item1[encoding[channel]["field"]] = round(
                                data_domain[0]
                                + (data_domain[1] - data_domain[0])
                                / (data_range[1] - data_range[0])
                                * (value - data_range[0]),
                                6,
                            )
                        else:  # temporal
                            item1[encoding[channel]["field"]] = round(
                                data_domain[0]
                                + (data_domain[1] - data_domain[0])
                                / (data_range[1] - data_range[0])
                                * (value - data_range[0]),
                                0,
                            )
                            # todo: extract date format
                            date_format = "%Y-%m-%d"
                            item1[encoding[channel]["field"]] = datetime.fromtimestamp(
                                item1[encoding[channel]["field"]]
                            ).strftime(date_format)
                    spec["data"].append(item1)

    if "text" in nodes:
        if len(nodes["text"]) == 1:
            spec["title"] = nodes["text"][0]["content"]


def analysis_scale(spec):
    encoding = spec["encoding"]
    for channel in encoding:
        scale = encoding[channel]["scale"]
        scale["ticks"] = scale["domain"]
        if "quantitative" == encoding[channel]["type"]:
            scale["domain"] = [float(value) for value in scale["domain"]]
            if "unit" in encoding[channel]:
                scale["domain"] = [
                    value * encoding[channel]["unit"] for value in scale["domain"]
                ]
                del encoding[channel]["unit"]
        elif "temporal" == encoding[channel]["type"]:
            # %Y-%m
            checks = [check_time_format(value, "%Y-%m") for value in scale["domain"]]
            if False not in checks:
                scale["domain"] = [(value + "-01") for value in scale["domain"]]
            scale["domain"] = [parse(value).timestamp() for value in scale["domain"]]
        if "nominal" != encoding[channel]["type"]:
            linear = True
            for i in range(2, len(scale["range"])):
                k1 = (scale["domain"][i] - scale["domain"][i - 1]) / (
                    scale["range"][i] - scale["range"][i - 1]
                )
                k2 = (scale["domain"][i - 1] - scale["domain"][i - 2]) / (
                    scale["range"][i - 1] - scale["range"][i - 2]
                )
                if abs((k1 - k2) / k1) > PRECISION:
                    linear = False
                    break
            if linear:
                scale["type"] = "linear"
                if len(scale["domain"]) > 1:
                    k = (scale["range"][-1] - scale["range"][0]) / (
                        scale["domain"][-1] - scale["domain"][0]
                    )
                    b = scale["range"][0] - k * scale["domain"][0]

                    if channel == "x":
                        if "x" in spec and "width" in spec:
                            scale["range"] = [spec["x"], spec["x"] + spec["width"]]
                            scale["domain"] = [
                                (scale["range"][0] - b) / k,
                                (scale["range"][1] - b) / k,
                            ]
                    elif channel == "y":
                        if "y" in spec and "height" in spec:
                            scale["range"] = [spec["y"] + spec["height"], spec["y"]]
                            scale["domain"] = [
                                (scale["range"][0] - b) / k,
                                (scale["range"][1] - b) / k,
                            ]
                max_index = scale["domain"].index(max(scale["domain"]))
                min_index = scale["domain"].index(min(scale["domain"]))
                scale["domain"] = [
                    scale["domain"][min_index],
                    scale["domain"][max_index],
                ]
                scale["range"] = [scale["range"][min_index], scale["range"][max_index]]
            # if not linear scale, maybe nominal
            if "type" not in scale:
                scale["domain"] = scale["ticks"]
                encoding[channel]["type"] = "nominal"


def get_leaf_nodes(spec):
    nodes = []
    if spec["tag"] == "g":
        for child in spec["children"]:
            nodes += get_leaf_nodes(child)
    else:
        nodes.append(spec)
    return nodes


def deconstruct(svg, source="matplotlib"):
    doc = minidom.parseString(svg)
    svg = doc.getElementsByTagName("svg")[0]

    # matplotlib parser
    defss = {}
    spec = parser_node(svg, None, defss, [0, 0], {}, source)
    subplots = [
        child
        for child in spec["children"][0]["children"]
        if ("type" in child and child["type"] == "subplot")
    ]
    if len(subplots) != 1:
        return None
    subplot = subplots[0]

    # find legend
    legends = [
        child
        for child in subplot["children"]
        if ("type" in child and child["type"] == "legend")
    ]
    legend = None
    if len(legends) > 1:
        return None
    elif len(legends) == 1:
        legend = legends[0]
    else:
        # legend may out of subplot
        legends = [
            child
            for child in spec["children"][0]["children"]
            if ("type" in child and child["type"] == "legend")
        ]
        if len(legends) == 1:
            legend = legends[0]

    subplot["encoding"] = {}
    if legend is not None:
        analysis_legend(legend, subplot["encoding"])
    others = {}
    for child in subplot["children"]:
        if "type" in child:
            if child["type"] == "xaxis" or child["type"] == "yaxis":
                analysis_axis(child, subplot["encoding"])
        else:
            nodes = get_leaf_nodes(child)
            for node in nodes:
                if node["tag"] not in others:
                    others[node["tag"]] = [node]
                else:
                    others[node["tag"]].append(node)
    analysis_scale(subplot)
    analysis_mark(others, subplot)
    return subplot
