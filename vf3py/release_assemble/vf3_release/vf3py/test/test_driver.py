def run_tests():
    from .basic_graphs import main as f_basic_graphs
    from .edge_test import main as f_edge_attrs
    from .mol_isoms import main as f_molecular
    from .mono import main as f_mono
    from .mono_random import main as f_mono_random
    from .simple_attrs import main as f_simple_attrs

    TEST_FUNCTIONS = {
        'basic_graphs': f_basic_graphs,
        'edge_attrs': f_edge_attrs,
        'molecular': f_molecular,
        'monomorphism': f_mono,
        'random monomorphism': f_mono_random,
        "Anton's example": f_simple_attrs,
    }

    for function_name, function in TEST_FUNCTIONS.items():
        print(f"=== Running test '{function_name}' ===")
        function()
    
    print("OK")


if __name__ == "__main__":
    run_tests()
