from vf3py_base import calc_noattrs, calc_nodeattr, calc_edgeattr, calc_bothattrs

# vf3py_base.calc([0, 1, 2, 3], [1, 1, 1, 1], [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], [], True, False)

res = calc_bothattrs(
    target={
        'nodes': [0, 1, 2, 3],
        'node_attrs': [1, 1, 1, 1],
        'edges': [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
        'edge_attrs': [1,1,1,1,1],
    },
    pattern={
        'nodes': [0, 1, 2],
        'node_attrs': [1, 1, 1],
        'edges': [[0, 1], [0, 2]],
        'edge_attrs': [1,1],
    }, directed=False, all_solutions=True, verbose=False
)
print(f"Result = {repr(res)}")

res = calc_noattrs(
    target={
        'nodes': [0, 1, 2, 3],
        'node_attrs': [],
        'edges': [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
        'edge_attrs': [],
    },
    pattern={
        'nodes': [0, 1, 2],
        'node_attrs': [],
        'edges': [[0, 1], [0, 2]],
        'edge_attrs': [],
    }, directed=False, all_solutions=True, verbose=False
)

print(f"Result = {repr(res)}")
