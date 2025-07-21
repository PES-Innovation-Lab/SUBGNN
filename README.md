# SUBGNN

#### Subgraph Matching using GNNs

## Step 1: Install Dependencies

```shell
pip install -r requirements.txt
```

## Step 2: Build Glasgow in `modified_glasgow`

```bash
cd modified_glasgow
cmake -S . -B ../build
cmake --build ../build
cd ..
```

## Step 3: Make python wrapper executable

```bash
chmod +x glasgow_subgraph_solver
```

## Step 4: Test using `start_from_vertex_example.py`

```bash
python start_form_vertex_example.py
```

## Can check `modified_glasgow/pythonTests` for extra options and how to use them.