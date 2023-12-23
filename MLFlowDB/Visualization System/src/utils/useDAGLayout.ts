import { dagConnect, sugiyama } from 'd3-dag'
import { useEffect } from 'react'
import { Edge, Position, ReactFlowState, useReactFlow, useStore } from 'reactflow'

// the layout direction (T = top, R = right, B = bottom, L = left, TB = top to bottom, ...)
export type Direction = 'TB' | 'LR' | 'RL' | 'BT'

export type Options = {
    direction: Direction
}

const positionMap: Record<string, Position> = {
    T: Position.Top,
    L: Position.Left,
    R: Position.Right,
    B: Position.Bottom,
}

const getPosition = (x: number, y: number, direction: Direction) => {
    switch (direction) {
        case 'LR':
            return { x: y, y: x }
        case 'RL':
            return { x: -y, y: -x }
        case 'BT':
            return { x: -x, y: -y }
        default:
            return { x, y }
    }
}

const creation = dagConnect()
// const DAGLayout = sugiyama().nodeSize((d) => {
//     console.log(d);

//     return [200, 300]
// })

export const useDAGLayout = ({ direction }: Options) => {
    const nodeCountSelector = (state: ReactFlowState) => state.nodeInternals.size
    const nodesInitializedSelector = (state: ReactFlowState) =>
        Array.from(state.nodeInternals.values()).every((node) => node.width && node.height)

    const nodeCount = useStore(nodeCountSelector)
    const nodesInitialized = useStore(nodesInitializedSelector)
    const { getEdges, getNodes, setNodes } = useReactFlow()

    useEffect(() => {
        if (!nodeCount || !nodesInitialized) {
            console.log("error!!")
            return
        }

        const nodes = getNodes()
        const nodesLookup = new Map(nodes.map(node => [node.id, {
            width: node.width ?? 0,
            height: node.height ?? 0,
        }]))

        const DAGLayout = sugiyama().nodeSize(d => {
            const data: { id: string } | undefined = d?.data
            if (!data) { return [0, 0] }
            const { id } = data
            const { width, height } = nodesLookup.get(id) || { width: 0, height: 0 }
            console.log(width, height);

            return [height * 2, width * 2]
        })


        const edges: Edge[] = getEdges().filter(e => e.data.category === 'node')

        const DAG = creation(edges.map(({ source, target }) => [source, target]))
        DAGLayout(DAG as any)

        const allNodes = DAG.descendants()

        const lookup = new Map(allNodes.map((node) => [node.data.id, node]))
        setNodes(nodes => nodes.map(node => {
            const { x, y } = lookup.get(node.id) || {}
            const { px, py } = { px: x ?? node.position.x, py: y ?? node.position.y }

            return {
                ...node,
                sourcePosition: positionMap[direction[1]],
                targetPosition: positionMap[direction[0]],
                position: getPosition(px, py, direction),
                style: { opacity: 1 },
            }
        }))

    }, [nodeCount, nodesInitialized, getEdges, setNodes, direction])

}
