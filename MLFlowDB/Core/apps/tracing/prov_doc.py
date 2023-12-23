import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List

import mongoengine
import pydot
from bson import ObjectId

from prov.constants import PROV_ATTRIBUTE_QNAMES
from prov.dot import ANNOTATION_END_ROW, ANNOTATION_ROW_TEMPLATE, ANNOTATION_START_ROW, ANNOTATION_LINK_STYLE
from prov.graph import INFERRED_ELEMENT_CLASS
from prov.model import (
    ProvEntity,
    ProvActivity,
    ProvAgent,
    ProvBundle,
    PROV_ACTIVITY,
    PROV_AGENT,
    PROV_ALTERNATE,
    PROV_ASSOCIATION,
    PROV_ATTRIBUTION,
    PROV_BUNDLE,
    PROV_COMMUNICATION,
    PROV_DERIVATION,
    PROV_DELEGATION,
    PROV_ENTITY,
    PROV_GENERATION,
    PROV_INFLUENCE,
    PROV_INVALIDATION,
    PROV_END,
    PROV_MEMBERSHIP,
    PROV_MENTION,
    PROV_SPECIALIZATION,
    PROV_START,
    PROV_USAGE,
    Identifier,
    PROV_ATTRIBUTE_QNAMES,
    sorted_attributes,
    ProvException,
)

from prov.graph import INFERRED_ELEMENT_CLASS
from prov.identifier import Identifier
from prov.model import ProvDocument, ProvException, sorted_attributes

from apps.model.mldr_model.ml_activity import HyperParameters
from apps.model.prov_dm import PROVObject, PROVRelation
from apps.tracing.base import TraceGraphNode, TraceGraphEdge

try:
    from html import escape
except ImportError:
    from cgi import escape

GENERIC_NODE_STYLE = {
    None: {
        "shape": "oval",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname" : "SimHei",
    },
    ProvEntity: {
        "shape": "oval",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname" : "SimHei",
    },
    ProvActivity: {
        "shape": "box",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname" : "SimHei",
    },
    ProvAgent: {
        "shape": "house",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname": "SimHei",
    },
    ProvBundle: {
        "shape": "folder",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname": "SimHei",
    },
}
DOT_PROV_STYLE = {
    # Generic node
    0: {
        "shape": "oval",
        "style": "filled",
        "fillcolor": "lightgray",
        "color": "dimgray",
        "fontname": "SimHei",
    },
    # Elements
    PROV_ENTITY: {
        "shape": "oval",
        "style": "filled",
        "fillcolor": "#FFFC87",
        "color": "#808080",
        "fontname": "SimHei",
    },
    PROV_ACTIVITY: {
        "shape": "box",
        "style": "filled",
        "fillcolor": "#9FB1FC",
        "color": "#0000FF",
        "fontname": "SimHei",
    },
    PROV_AGENT: {"shape": "house", "style": "filled", "fillcolor": "#FED37F","fontname": "SimHei"},
    PROV_BUNDLE: {"shape": "folder", "style": "filled", "fillcolor": "aliceblue","fontname": "SimHei"},
    # Relations
    PROV_GENERATION: {
        "label": "wasGeneratedBy",
        "fontsize": "10.0",
        "color": "darkgreen",
        "fontcolor": "darkgreen",
        "fontname": "SimHei",
    },
    PROV_USAGE: {
        "label": "used",
        "fontsize": "10.0",
        "color": "red4",
        "fontcolor": "red",
        "fontname": "SimHei",
    },
    PROV_COMMUNICATION: {"label": "wasInformedBy", "fontsize": "10.0","fontname": "SimHei"},
    PROV_START: {"label": "wasStartedBy", "fontsize": "10.0","fontname": "SimHei"},
    PROV_END: {"label": "wasEndedBy", "fontsize": "10.0","fontname": "SimHei"},
    PROV_INVALIDATION: {"label": "wasInvalidatedBy", "fontsize": "10.0","fontname": "SimHei"},
    PROV_DERIVATION: {"label": "wasDerivedFrom", "fontsize": "10.0","fontname": "SimHei"},
    PROV_ATTRIBUTION: {
        "label": "wasAttributedTo",
        "fontsize": "10.0",
        "color": "#FED37F",
        "fontname": "SimHei",
    },
    PROV_ASSOCIATION: {
        "label": "wasAssociatedWith",
        "fontsize": "10.0",
        "color": "#FED37F",
        "fontname": "SimHei",
    },
    PROV_DELEGATION: {
        "label": "actedOnBehalfOf",
        "fontsize": "10.0",
        "color": "#FED37F",
        "fontname": "SimHei",
    },
    PROV_INFLUENCE: {"label": "wasInfluencedBy", "fontsize": "10.0", "color": "grey","fontname": "SimHei"},
    PROV_ALTERNATE: {"label": "alternateOf", "fontsize": "10.0","fontname": "SimHei"},
    PROV_SPECIALIZATION: {"label": "specializationOf", "fontsize": "10.0","fontname": "SimHei"},
    PROV_MENTION: {"label": "mentionOf", "fontsize": "10.0","fontname": "SimHei"},
    PROV_MEMBERSHIP: {"label": "hadMember", "fontsize": "10.0","fontname": "SimHei"},
}
ANNOTATION_STYLE = {
    "shape": "note",
    "color": "gray",
    "fontcolor": "black",
    "fontsize": "10",
    "fontname": "SimHei",
}

def prov_to_dot(
    bundle,
    show_nary=True,
    use_labels=False,
    direction="BT",
    show_element_attributes=True,
    show_relation_attributes=True,
    charset="utf-8"
):
    """
    Convert a provenance bundle/document into a DOT graphical representation.

    :param bundle: The provenance bundle/document to be converted.
    :type bundle: :class:`ProvBundle`
    :param show_nary: shows all elements in n-ary relations.
    :type show_nary: bool
    :param use_labels: uses the prov:label property of an element as its name (instead of its identifier).
    :type use_labels: bool
    :param direction: specifies the direction of the graph. Valid values are "BT" (default), "TB", "LR", "RL".
    :param show_element_attributes: shows attributes of elements.
    :type show_element_attributes: bool
    :param show_relation_attributes: shows attributes of relations.
    :type show_relation_attributes: bool
    :returns:  :class:`pydot.Dot` -- the Dot object.
    """
    if direction not in {"BT", "TB", "LR", "RL"}:
        # Invalid direction is provided
        direction = "BT"  # reset it to the default value
    maindot = pydot.Dot(graph_type="digraph", rankdir=direction, charset=charset,)

    node_map = {}
    count = [0, 0, 0, 0]  # counters for node ids

    def _bundle_to_dot(dot, bundle):
        def _attach_attribute_annotation(node, record):
            # Adding a node to show all attributes
            attributes = list(
                (attr_name, value)
                for attr_name, value in record.attributes
                if attr_name not in PROV_ATTRIBUTE_QNAMES
            )

            if not attributes:
                return  # No attribute to display

            # Sort the attributes.
            attributes = sorted_attributes(record.get_type(), attributes)

            ann_rows = [ANNOTATION_START_ROW]
            ann_rows.extend(
                ANNOTATION_ROW_TEMPLATE
                % (
                    attr.uri,
                    escape(str(attr)),
                    ' href="%s"' % value.uri if isinstance(value, Identifier) else "",
                    escape(
                        str(value)
                        if not isinstance(value, datetime)
                        else str(value.isoformat())
                    ),
                )
                for attr, value in attributes
            )
            ann_rows.append(ANNOTATION_END_ROW)
            count[3] += 1
            annotations = pydot.Node(
                "ann%d" % count[3], label="\n".join(ann_rows), **ANNOTATION_STYLE
            )
            dot.add_node(annotations)
            dot.add_edge(pydot.Edge(annotations, node, **ANNOTATION_LINK_STYLE))

        def _add_bundle(bundle):
            count[2] += 1
            subdot = pydot.Cluster(
                graph_name="c%d" % count[2], URL=f'"{bundle.identifier.uri}"'
            )
            if use_labels:
                if bundle.label == bundle.identifier:
                    bundle_label = f'"{bundle.label}"'
                else:
                    # Fancier label if both are different. The label will be the main
                    # node text, whereas the identifier will be a kind of subtitle.
                    bundle_label = (
                        f"<{bundle.label}<br />"
                        f'<font color="#333333" point-size="10">'
                        f'{bundle.identifier}</font>>'
                    )
                subdot.set_label(f'"{bundle_label}"')
            else:
                subdot.set_label('"%s"' % str(bundle.identifier))
            _bundle_to_dot(subdot, bundle)
            dot.add_subgraph(subdot)
            return subdot

        def _add_node(record):
            count[0] += 1
            node_id = "n%d" % count[0]
            if use_labels:
                if record.label == record.identifier:
                    node_label = f'"{record.label}"'
                else:
                    # Fancier label if both are different. The label will be
                    # the main node text, whereas the identifier will be a
                    # kind of subtitle.
                    node_label = (
                        f"<{record.label}<br />"
                        f'<font color="#333333" point-size="10">'
                        f'{record.identifier}</font>>'
                    )
            else:
                node_label = f'"{record.identifier}"'

            uri = record.identifier.uri
            style = DOT_PROV_STYLE[record.get_type()]
            node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
            node_map[uri] = node
            dot.add_node(node)

            if show_element_attributes:
                _attach_attribute_annotation(node, rec)
            return node

        def _add_generic_node(qname, prov_type=None):
            count[0] += 1
            node_id = "n%d" % count[0]
            node_label = f'"{qname}"'

            uri = qname.uri
            style = GENERIC_NODE_STYLE[prov_type] if prov_type else DOT_PROV_STYLE[0]
            node = pydot.Node(node_id, label=node_label, URL='"%s"' % uri, **style)
            node_map[uri] = node
            dot.add_node(node)
            return node

        def _get_bnode():
            count[1] += 1
            bnode_id = "b%d" % count[1]
            bnode = pydot.Node(bnode_id, label='""', shape="point", color="gray")
            dot.add_node(bnode)
            return bnode

        def _get_node(qname, prov_type=None):
            if qname is None:
                return _get_bnode()
            uri = qname.uri
            if uri not in node_map:
                _add_generic_node(qname, prov_type)
            return node_map[uri]

        records = bundle.get_records()
        relations = []
        for rec in records:
            if rec.is_element():
                _add_node(rec)
            else:
                # Saving the relations for later processing
                relations.append(rec)

        if not bundle.is_bundle():
            for bundle in bundle.bundles:
                _add_bundle(bundle)

        for rec in relations:
            args = rec.args
            # skipping empty records
            if not args:
                continue
            # picking element nodes
            attr_names, nodes = zip(
                *(
                    (attr_name, value)
                    for attr_name, value in rec.formal_attributes
                    if attr_name in PROV_ATTRIBUTE_QNAMES
                )
            )
            inferred_types = list(map(INFERRED_ELEMENT_CLASS.get, attr_names))
            other_attributes = [
                (attr_name, value)
                for attr_name, value in rec.attributes
                if attr_name not in PROV_ATTRIBUTE_QNAMES
            ]
            add_attribute_annotation = show_relation_attributes and other_attributes
            add_nary_elements = len(nodes) > 2 and show_nary
            style = DOT_PROV_STYLE[rec.get_type()]
            if len(nodes) < 2:  # too few elements for a relation?
                continue  # cannot draw this

            if add_nary_elements or add_attribute_annotation:
                # a blank node for n-ary relations or the attribute annotation
                bnode = _get_bnode()

                # the first segment
                dot.add_edge(
                    pydot.Edge(
                        _get_node(nodes[0], inferred_types[0]),
                        bnode,
                        arrowhead="none",
                        **style,
                    )
                )
                style = dict(style)  # copy the style
                del style["label"]  # not showing label in the second segment
                # the second segment
                dot.add_edge(
                    pydot.Edge(bnode, _get_node(nodes[1], inferred_types[1]), **style)
                )
                if add_nary_elements:
                    style["color"] = "gray"  # all remaining segment to be gray
                    style["fontcolor"] = "dimgray"  # text in darker gray
                    for attr_name, node, inferred_type in zip(
                        attr_names[2:], nodes[2:], inferred_types[2:]
                    ):
                        if node is not None:
                            style["label"] = attr_name.localpart
                            dot.add_edge(
                                pydot.Edge(
                                    bnode, _get_node(node, inferred_type), **style
                                )
                            )
                if add_attribute_annotation:
                    _attach_attribute_annotation(bnode, rec)
            else:
                # show a simple binary relations with no annotation
                dot.add_edge(
                    pydot.Edge(
                        _get_node(nodes[0], inferred_types[0]),
                        _get_node(nodes[1], inferred_types[1]),
                        **style,
                    )
                )

    try:
        unified = bundle.unified()
    except ProvException:
        # Could not unify this bundle
        # try the original document anyway
        unified = bundle

    _bundle_to_dot(maindot, unified)
    return maindot

class ProvDoc:
    @staticmethod
    def generate_graph(nodes:Dict[str,TraceGraphNode],edges:List[TraceGraphEdge]):
        doc = ProvDocument()
        doc.add_namespace('mldr', 'http://mged.ustb.edu.cn')

        for id,node in nodes.items():
            if node.prov_type == PROVObject.ENTITY:
                doc.entity(identifier=node.node_with_namespace)
            elif node.prov_type == PROVObject.ACTIVITY:
                doc.activity(identifier=node.node_with_namespace)
            elif node.prov_type == PROVObject.AGENT:
                doc.agent(identifier=node.node_with_namespace)
            else:
                raise NotImplementedError("no such type in PROV-DM!")
        for edge in edges:
            if edge.type == PROVRelation.GENERATION:
                doc.wasGeneratedBy(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.USE:
                doc.used(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.DERIVATION:
                doc.wasDerivedFrom(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.ATTRIBUTE:
                doc.wasAttributedTo(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.QUOTE:
                doc.wasQuotedFrom(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.MEMBER:
                doc.hadMember(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
            elif edge.type == PROVRelation.INFORM:
                doc.wasInformedBy(nodes[edge.started].node_with_namespace, nodes[edge.ended].node_with_namespace)
        dot = prov_to_dot(doc,charset='utf-8')

        prov_graph_name = f"prov_{str(uuid.uuid1())}.png"
        dot.write_png(prov_graph_name)

        return prov_graph_name

class MLDRPhase(Enum):
    DATA_GATHERING = 'data-gathering'
    DATA_TRANSFORMING = 'data-transforming'
    DATASET = 'dataset'
    ML_ACTIVITY = 'ml-activity'
    ML_ARCHVING = 'ml-archving'

class ProvOnline:
    @staticmethod
    def _generate_edge(edge:TraceGraphEdge):
        return {
            "type": edge.type,
            "started": str(edge.started),
            "ended": str(edge.ended)
        }

    @staticmethod
    def _generate_for_base(node:TraceGraphNode,res):
        expect_keys = ["id","name","name_temp","phase","type","mldr_type","version"]

        new_infos = {}
        infos = {key: value for key, value in res.items() if key not in expect_keys}
        for key,item in infos.items():
            if isinstance(item,HyperParameters):
                new_infos[key] = item.to_dict()
            else:
                new_infos[key] = str(item)

        return {
            "id": res.get("id"),
            "name": node.get_name(),
            "phase": res.get("phase"),
            "mldr_type": res.get("mldr_type"),
            "type": res.get("type"),
            "info": new_infos,
            "version": res.get("version",1)
        }

    @staticmethod
    def _generate_for_entity(node:TraceGraphNode,res):
        return {**ProvOnline._generate_for_base(node,res),"stored_data": {}}
    @staticmethod
    def _generate_for_activity(node:TraceGraphNode,res):
        return {**ProvOnline._generate_for_base(node,res),"mapping_data": {}}
    @staticmethod
    def _generate_for_agent(node:TraceGraphNode,res):
        return {**ProvOnline._generate_for_base(node,res),"agent_data": {}}

    @staticmethod
    def _handle_property_dict(res:Dict):
        def _get_pahse(mldr_type:str):
            if mldr_type.startswith("DataPreparation"):
                return MLDRPhase.DATA_TRANSFORMING
            elif mldr_type.startswith("MachineLearningModel"):
                return MLDRPhase.ML_ARCHVING
            elif mldr_type.startswith("OriginData"):
                return MLDRPhase.DATA_GATHERING
            elif mldr_type.startswith("MachineLearning"):
                return MLDRPhase.ML_ACTIVITY
            elif mldr_type.startswith("Dataset"):
                return MLDRPhase.DATASET
            raise ValueError

        res_new = {}
        for key,item in res.items():
            if issubclass(type(item),Enum):
                res_new[key] = item.value
            elif isinstance(item,ObjectId):
                if key == "id":
                    res_new[key] = str(item)
                else:
                    continue
            elif isinstance(item,datetime):
                res_new[key] = item.strftime("%Y-%m-%d %H:%M:%S")
            elif key=="_cls":
                res_new["mldr_type"] = str(item).split(".")[-1]
                res_new["phase"] = _get_pahse(res_new["mldr_type"]).value
            elif key == "artifact_type":
                res_new["type"] = item.split(":")[-1]
            else:
                res_new[key] = item
        return res_new

    @staticmethod
    def generate_graph(nodes: Dict[str, TraceGraphNode], edges: List[TraceGraphEdge]):
        res_nodes = []
        for id,node in nodes.items():
            data = ProvOnline._handle_property_dict(node.object.to_property_dict())
            if data.get("type") == "entity":
                res_nodes.append(ProvOnline._generate_for_entity(node,data))
            elif data.get("type") == "agent":
                res_nodes.append(ProvOnline._generate_for_agent(node,data))
            elif data.get("type") == "activity":
                res_nodes.append(ProvOnline._generate_for_activity(node,data))
            else:
                print(data.get("type"))
                raise ValueError

        res_edges = []
        for edge in edges:
            res_edges.append(ProvOnline._generate_edge(edge))

        return res_nodes,res_edges
