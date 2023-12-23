import { NodeData } from "@/utils/process"
import { PropsWithChildren } from "react"
import { Handle, NodeProps, Position } from "reactflow"
import internal from "stream";
import {stringify} from "postcss";

export { default as EntityNode } from './entity'
export { default as ActivityNode } from './activity'
export { default as AgentNode } from './agent'

type WorkflowNodeProps<T extends NodeData> = NodeProps<T> & PropsWithChildren<{
    id: string
    backgroundColor?: string,
    borderRadius?: string,
    objectTransform?:string,
    contentTransform?:string,
}>

export default function Node<T extends NodeData>({ children, id, data, selected, backgroundColor,borderRadius,objectTransform,contentTransform }: WorkflowNodeProps<T>) {
    const { name,mldr_type,type } = data
    return <>

        <div
            style={{ backgroundColor: backgroundColor ?? "#fff",borderRadius:borderRadius ?? "5px",transform:objectTransform ?? "", border:'1px solid #00000f' }}
            className={`
            flex flex-col
            ${selected && `outline outline-2 outline-gray-600`}
            transition-shadow shadow-lg ${selected && `shadow-xl`}
            w-[250px] cursor-pointer
            `}
        >
            <div style={{transform:contentTransform ?? "" }}>
                {/*<div style={{ fontSize:"10px" }} className="h-[15px] flex justify-center items-center">{"<prov::"+type+">"}</div>*/}
                <div style={{ fontSize:"10px" }} className="h-[15px] flex justify-center items-center">{"<mldr::"+mldr_type+">"}</div>
                <div className="h-[30px] flex justify-center items-center">{name}</div>

                <div className="flex flex-col space-y-2">
                    {children}
                </div>
            </div>
        </div>

        <Handle type="source" style={{visibility: "hidden"}} position={Position.Left} id={id} />
        <Handle type="target" style={{visibility: "hidden"}} position={Position.Right} id={id} />
        {/*<Handle type="target" style={{visibility: "hidden"}} position={Position.Top} id={id} />*/}
        {/*<Handle type="source" style={{visibility: "hidden"}} position={Position.Bottom} id={id} />*/}
    </>
}
