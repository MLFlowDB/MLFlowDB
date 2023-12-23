import {AttrID, EntityNodeData, backgroundColor} from "@/utils/process"
import {Handle, NodeProps} from "reactflow"
import Node from '.'
import Attribute from "./attribute"
import React from "react";

export type EntityNodeProps = NodeProps<EntityNodeData>

export default (props: NodeProps<EntityNodeData>) => {
    const { id,data: { name,phase, stored_data ,mldr_type,type} } = props
    return <Node<EntityNodeData> {...props} backgroundColor={backgroundColor[phase]}>
        {Object.entries(stored_data ?? {})
            .map(([name, value]) =>
                <Attribute
                    key={AttrID(id, name)}
                    sourceID={AttrID(id, name)}
                    targetID={AttrID(id, name)}
                    name={name}
                    value={value} />)}
    </Node>

}