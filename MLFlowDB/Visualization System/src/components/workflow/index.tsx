import { EntityNode, ActivityNode,AgentNode } from '@/components/node'
import process, { rawNodes,rawEdges } from '@/utils/process'
import ReactFlow, {
    MiniMap,
    Background,
    BackgroundVariant,
    Controls,
    NodeTypes,
    useEdgesState,
    useNodesState,
    useReactFlow, useOnSelectionChange
} from 'reactflow'
import { Direction, useDAGLayout } from '@/utils/useDAGLayout'
import axios from 'axios';
import {useCallback, useEffect} from "react";


const nodeTypes: NodeTypes = {
    entity: EntityNode,
    activity: ActivityNode,
    agent: AgentNode,
}


export type WorkflowProps = {
    datas : object
    direction?: Direction
}

const minimapStyle = {
    height: 120,
};


const { nodes: initNodes, edges: initEdges } = process(rawNodes,rawEdges)

export default ({ direction = 'RL',datas }: WorkflowProps) => {
    useDAGLayout({ direction })

    const [nodes, _setNodes, onNodesChange] = useNodesState(initNodes)
    const [edges, _setEdges, onEdgesChange] = useEdgesState(initEdges)

    const { fitView } = useReactFlow();

    // every time our nodes change, we want to center the graph again
    useEffect(() => {
        fitView({ duration: 400 });
    }, [nodes, fitView]);

    useEffect(() => {
        // @ts-ignore
        const { nodes: newNodes, edges: newEdges } = process(datas.nodes, datas.edges);
        _setNodes(newNodes);
        _setEdges(newEdges);
    }, [datas]);



    return <div className='w-full h-full'>
        <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            fitView
            // minZoom={0.5}
            // maxZoom={3}
            proOptions={{ hideAttribution: true }}
            nodeTypes={nodeTypes}>
            <Controls />
            <MiniMap style={minimapStyle} />
        </ReactFlow>
    </div>
}