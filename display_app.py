import streamlit as st
import itertools
import os

# Define hyperparameter variations
learning_rates = [0.001, 0.005, 0.01, 0.1]
dropouts = [0.1, 0.2, 0.3, 0.5, 1.0]
gradient_clip_vals = [0.2, 0.5, 0.7, 1, 2, 3, 10]
hidden_sizes = [8, 16, 32, 64]
hidden_continuous_sizes = [8, 16, 32, 64]
attention_head_sizes = [1, 2, 3, 4]

# Generate all combinations (Cartesian product)
combinations = list(itertools.product(
    learning_rates,
    dropouts,
    gradient_clip_vals,
    hidden_sizes,
    hidden_continuous_sizes,
    attention_head_sizes
))

st.title("Hyperparameter Exploration -- Categorical ID=50_new")

# Create columns to display the dropdowns side by side
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    selected_lr = st.selectbox("LR", learning_rates)

with col2:
    selected_dropout = st.selectbox("Dropout", dropouts)

with col3:
    selected_gc = st.selectbox("Grad Clip", gradient_clip_vals)

with col4:
    selected_hs = st.selectbox("Hidden Size", hidden_sizes)

with col5:
    selected_hcs = st.selectbox("Hidden Cont. Size", hidden_continuous_sizes)

with col6:
    selected_ah = st.selectbox("Attn Heads", attention_head_sizes)

# Find the index in `combinations` that matches the user selection
selected_index = None
for i, (lr, do, gc, hs, hcs, ah) in enumerate(combinations):
    if (
        lr == selected_lr and
        do == selected_dropout and
        gc == selected_gc and
        hs == selected_hs and
        hcs == selected_hcs and
        ah == selected_ah
    ):
        selected_index = i
        break

# Display the chosen combination and index
if selected_index is not None:
    st.write(f"You selected combination index: {selected_index}")
    st.write(f"LR={selected_lr}, Dropout={selected_dropout}, "
             f"GC={selected_gc}, HS={selected_hs}, "
             f"HCS={selected_hcs}, AttnHeads={selected_ah}")

    # Build the path to the corresponding plot (e.g. 'plots/loss_plot_0.png')
    plot_path = os.path.join("plots", f"loss_plot_{selected_index}.png")

    # Display the image, if it exists
    if os.path.isfile(plot_path):
        st.image(plot_path, caption=f"Loss plot for combination {selected_index}")
    else:
        st.warning(f"Plot not found: {plot_path}")
else:
    st.warning("No valid combination found (shouldn't happen with given lists).")
