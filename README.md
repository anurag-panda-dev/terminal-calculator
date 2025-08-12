# Terminal Calculator

A **simple yet powerful terminal-based calculator** written in Python.  
Created by **Anurag Panda**  
Date: 12-08-2025

---

## Features

This calculator supports:

1. **Simple Calculations**  
   - Evaluate arithmetic expressions (e.g., `2+3*4`, `sin(pi/4)`, `(x+1)**2` with variable input).
2. **Derivatives Calculation**  
   - Compute symbolic derivatives of functions (with optional evaluation at a point).
3. **Integration Calculation**  
   - Compute symbolic indefinite or definite integrals.
4. **Trigonometry Calculation**  
   - Perform trigonometric and inverse trigonometric calculations (degrees/radians).
5. **Number System Conversion**  
   - Convert integers between bases (2 to 36).
6. **2D Graph Plotting**  
   - Plot functions `y = f(x)` over a specified range.
7. **3D Graph Plotting**  
   - Plot surfaces `z = f(x, y)` over specified ranges.

---

## Requirements

- Python 3.8+
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [sympy](https://www.sympy.org/)

Install dependencies with:

```sh
pip install -r requirements.txt
```

---

## Usage

Run the calculator from the terminal:

```sh
python terminal-calculator.py
```

You will see a menu with options. Enter the number corresponding to the feature you want to use.

---

## Example Menu

```
===== Terminal Calculator Menu =====
1) Simple calculation (expressions)
2) Derivative calculation (symbolic)
3) Integration calculation (symbolic)
4) Trigonometry calculations
5) Number system conversion (integers)
6) 2D graph plotting (y = f(x))
7) 3D graph plotting (z = f(x,y))
0) Exit
====================================
```

---

## Notes

- **Expressions**: Use Python-style math (e.g., `**` for powers, `sin(x)`, `cos(x)`, etc.).
- **Variables**: For expressions with variables, you'll be prompted to enter their values.
- **Plotting**: Close the plot window to return to the menu.
- **Number Conversion**: Only integer conversions are supported.

---

## License

This project is for educational purposes.

---

## Author

Anurag Panda
