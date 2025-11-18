# Minimal Fixes for ThunderKittens Type Errors

This document explains the **three key fixes** needed to make your chunked attention kernel compile with ThunderKittens.

## Fix #1: Wrap Raw Pointers in `gl<>` Layout

**Error:**
```
error: no instance of overloaded function "kittens::group::load" matches the argument list
argument types are: (kittens::st<float, 64, 128>, float *__restrict__, {...})
```

**Problem:**
You were trying to load from raw `float*` pointers:
```cpp
auto* __restrict__ k = reinterpret_cast<float*>(g.gKeys[chunk_idx]);
warp::load(Ks, k, {head_idx, 0, 0});  // ERROR: k is raw pointer
```

**Fix:**
Wrap the pointer in a `gl<>` (global layout) descriptor:
```cpp
auto* __restrict__ k_ptr = reinterpret_cast<scalar_t*>(g.gKeys[chunk_idx]);
using kv_gl = gl<scalar_t, 1, -1, chunk_size, d_head>;
kv_gl k{k_ptr, nullptr, g.n_heads, chunk_size, nullptr};
warp::load(Ks, k, {0, head_idx, k_start, 0});  // WORKS!
```

**Why:** TK's load operations need the global layout to know dimensions and strides.

---

## Fix #2: Use Proper Vector Types for Row Operations

**Error:**
```
error: static assertion failed
static_assert(std::is_same_v<typename V::layout, typename rt_base<...>::col_vec_layout>);
```

**Problem:**
You were using `rv<float, N, naive>` for row reduction outputs:
```cpp
rv<float, 16, naive> max_vec;
warp::row_max(max_vec, output_reg);  // ERROR: wrong layout type
```

**Fix:**
Use the register tile's `col_vec` type:
```cpp
using vec_type = typename rt<scalar_t, TILE_SIZE<d_head>, TILE_SIZE<d_head>>::col_vec;
vec_type max_vec;
warp::row_max(max_vec, output_reg);  // WORKS!
```

**Why:** For row operations on `rt<T, rows, cols, row_layout>`, the result must be `rv<T, rows, ortho_layout>`, not `naive`. The `rt::col_vec` typedef gives you the correct type automatically.

---

## Fix #3: Use `mma_ABt` and `mma_AB` (not `mm_*`)

**Error:**
```
error: no instance of overloaded function "kittens::group::mm_ABt" matches the argument list
```

**Problem:**
You were using function names that don't exist:
```cpp
kittens::warp::mm_ABt(output_reg, Qr, Kr);   // ERROR: no such function
kittens::warp::mm_AB(result, output_reg, Vr); // ERROR: no such function
```

**Fix:**
Use the correct TK function names:
```cpp
warp::mma_ABt(output_reg, Qr, Kr);      // WORKS!
warp::mma_AB(result, output_reg, Vr);   // WORKS!
```

**Why:** ThunderKittens uses `mma_AB` and `mma_ABt` (matrix multiply-accumulate) for these operations.

---

## Summary

| What You Had | What You Need | Why |
|-------------|---------------|-----|
| `warp::load(Ks, k_ptr, {...})` | `warp::load(Ks, k_gl, {...})` | TK needs `gl<>` wrapper |
| `rv<float, N, naive> max_vec` | `typename rt<...>::col_vec max_vec` | TK enforces correct layout types |
| `warp::mm_ABt(...)` | `warp::mma_ABt(...)` | Correct TK function name |
| `warp::mm_AB(...)` | `warp::mma_AB(...)` | Correct TK function name |

These are the **only changes needed** to fix the type errors. Your kernel structure and algorithm stay exactly the same.
