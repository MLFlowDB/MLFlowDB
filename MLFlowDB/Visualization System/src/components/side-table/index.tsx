import {useState} from 'react';
import ReactFlow, {useOnSelectionChange} from 'reactflow';
import {ML_Phase, PROV_DM} from "@/utils/process";

type BlockProps = {
    title: string,
    phase: ML_Phase,
    mldr_type: string,
    version:number,
    prov_type: PROV_DM,
    attributes: { [name: string]: string }
}
const Block = ({title, attributes,phase, mldr_type, prov_type,version}: BlockProps) => {

    return <div className="w-full min-h-[20px] shadow-md rounded-lg border border-gray-200 p-2">
        <div  className="font-semibold mb-2 text-lg">{title}</div>
        <div className="min-h-[20px] text-xs"><a className="font-semibold">{"PROV Type : "}</a>  {"prov::" + prov_type}</div>
        <div className="min-h-[20px] text-xs"><a className="font-semibold">{"MLDR Type : "}</a> {"mldr::" + mldr_type}</div>
        <div className="min-h-[20px] text-xs"><a className="font-semibold">{"Phase : "}</a> {phase}</div>
        <div className="min-h-[20px] text-xs"><a className="font-semibold">{"Version : "}</a> {version}</div>

        {Object.keys(attributes).length > 0 && (
        <div className="w-full min-h-[20px] shadow-md rounded-lg border border-gray-200 p-2 text-xs">
            <div  className="font-semibold mb-2">{"Attributes"}</div>
        {Object.entries(attributes).map(([name, value]) =>
            <div key={name} className="flex flex-row mb-2 space-x-1 overflow-hidden">
                <div className="min-h-[20px] text-xs"><a className="font-semibold">{name} : </a>
                    {(typeof value != 'object') && (value)}
                    {   // @ts-ignore
                        (typeof value == 'object') && (Object.entries(value).map(([k, v]) => <div><a>{k}</a> : <a>{v}</a></div>))
                    }
                </div>
            </div>)}
        </div>)}

    </div>
}

// const init = [
//     {
//         title: '节点属性1',
//         attributes: {
//             '属性1': '值1',
//             '属性2': '值2',
//             '属性3': '值3',
//         }
//     },
//     {
//         title: '节点属性2',
//         attributes: {
//             '属性1': '值1',
//             '属性2': '值2',
//             '属性3': '值3',
//         }
//     },
//     {
//         title: '节点属性3',
//         attributes: {
//             '属性1': '值1',
//             '属性2': '值2',
//             '属性3': '值3',
//         }
//     },
//     {
//         title: '节点属性4',
//         attributes: {
//             '属性1': '值1',
//             '属性2': '值2',
//             '属性3': '值3',
//         }
//     },
// ]

export default () => {

    const [selectedNodes, setSelectedNodes] = useState([]);
    const [selectedEdges, setSelectedEdges] = useState([]);

    useOnSelectionChange({
        onChange: ({nodes, edges}) => {
            // @ts-ignore
            setSelectedNodes(nodes.map((node) => node));
            // @ts-ignore
            setSelectedEdges(edges.map((edge) => edge));
        },
    });

    const infos: BlockProps[] = []
    for (const singleNode of selectedNodes) {
        const {data} = singleNode
        // @ts-ignore
        infos.push({title: data.name, attributes: data.info ?? {}, mldr_type: data.mldr_type, prov_type: data.type, phase: data.phase,version:data.version})
    }

    console.log(infos)

    return <div>{Object.keys(infos).length > 0 && (
        <div className="w-full h-full space-y-2 p-2 overflow-y-scroll">
        {infos.map((block, index) => <Block title={block.title} attributes={block.attributes} key={index}
                                            mldr_type={block.mldr_type} prov_type={block.prov_type} phase={block.phase} version={block.version}/>)}
        </div>
    )}</div>
}