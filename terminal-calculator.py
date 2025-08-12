"""
Created By Anurag Panda
date: 12-08-2025 (20:00)


A simple terminal calculator
that performs: 
Simple calculation 
Derivatives calculation 
Integration calculation 
Trigonometry calculation 
Number system conversion 
2d graph plotting 
3d graph plotting
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

#----------helper utilities------------
def sanitize_input_expr(exp_str: str) -> str:
    """sanitize input expression"""
    # replace '^' with '**'
    return exp_str.strip().replace('^', '**')

def parse_sympy_expr(expr_str: str, symbols=None):
    """Parse a string expression into a sympy expression."""
    expr_str = sanitize_input_expr(expr_str)
    try:
        expr = sp.sympify(expr_str)
        return expr
    except Exception as e:
        raise ValueError(f"Invalid expression: {expr_str}") from e
    
def float_input(prompt: str, default=None):
    while True:
        raw = input(prompt).strip()
        if raw == '' and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Invalid input. Please enter a number.")

#------------------feature implementation----------------
def simple_calculation():
    print("\nsimple calulation (example: 2+3*4, sin(pi/4), (x+1)**2 with x=2)...")
    expr_str = input("Enter expression: ").strip()
    
    if not expr_str:
        print("No expression provided.")
        return
    
    expr_str = sanitize_input_expr(expr_str)
    
    try:
        expr = parse_sympy_expr(expr_str)
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        subs = {}
        
        if symbols:
            print("Detected symbols: ", ",".join(str(s) for s in symbols))
            for s in symbols:
                val = input(f"Enter value for {s} (or press Enter to skip): ").strip()
                
                if val != '':
                    try:
                        subs[s] = sp.sympify(val)
                        
                    except Exception:
                        subs[s] = float(val)
        result = expr.evalf(subs=subs) if subs else expr.evalf()
        print("Result: ", result)
    except Exception as e:
        print("Error: ", e)

def derivative_calculation():
    """
    Compute symbolic derivative(s) of a function w.r.t a variable and optional evaluation.
    """
    print("\nDerivative calculation")
    func_str = input("Enter function (in variable x, e.g. sin(x) + x**2): ").strip()
    if not func_str:
        print("No function entered. Returning to menu.")
        return
    func_str = sanitize_input_expr(func_str)
    try:
        x = sp.Symbol('x')  # default variable
        f = parse_sympy_expr(func_str)
        # Ask order of derivative
        order_raw = input("Derivative order (default 1): ").strip()
        order = int(order_raw) if order_raw else 1
        # Compute derivative symbolically
        deriv = sp.diff(f, x, order)
        print(f"\nSymbolic derivative (order {order}):")
        print(deriv)
        # Optionally evaluate at a point
        eval_at = input("Evaluate derivative at x = ? (leave blank to skip): ").strip()
        if eval_at != '':
            val = sp.N(deriv.subs(x, sp.sympify(eval_at)))
            print(f"Derivative at x = {eval_at} is {val}")
    except Exception as e:
        print("Error computing derivative:", e)

def integration_calculation():
    """
    Compute symbolic indefinite or definite integral of a function.
    """
    print("\nIntegration calculation")
    func_str = input("Enter function to integrate (in variable x): ").strip()
    if not func_str:
        print("No function entered. Returning to menu.")
        return
    func_str = sanitize_input_expr(func_str)
    try:
        x = sp.Symbol('x')
        f = parse_sympy_expr(func_str)
        integral_type = input("Type of integration - (I)ndefinite or (D)efinite? [I/D] (default I): ").strip().lower()
        if integral_type == 'd':
            # definite: ask limits
            a = input("Lower limit a: ").strip()
            b = input("Upper limit b: ").strip()
            if a == '' or b == '':
                print("Limits required for definite integral. Returning.")
                return
            A = sp.sympify(a)
            B = sp.sympify(b)
            res = sp.integrate(f, (x, A, B))
            print(f"\nDefinite integral from {A} to {B}:")
            print(res)
            try:
                # numeric value
                print("Numeric approximation:", sp.N(res))
            except Exception:
                pass
        else:
            # indefinite integral
            res = sp.integrate(f, x)
            print("\nIndefinite integral (plus C):")
            print(res)
    except Exception as e:
        print("Error computing integral:", e)

def trigonometry_calculation():
    """
    Perform basic trigonometric calculations with degrees/radians choice.
    """
    print("\nTrigonometry calculations")
    print("Options: sin, cos, tan, asin, acos, atan")
    func = input("Choose function (e.g. sin): ").strip().lower()
    if func not in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan'):
        print("Unsupported trig function. Returning.")
        return
    unit = input("Input unit - (D)egrees or (R)adians? [D/R] (default D): ").strip().lower()
    is_deg = (unit != 'r')
    val_raw = input("Enter value (angle for sin/cos/tan or value for inverse functions): ").strip()
    if val_raw == '':
        print("No value entered. Returning.")
        return
    try:
        v = float(val_raw)
        # convert to radians if needed for direct trig
        if func in ('sin', 'cos', 'tan'):
            arg = math.radians(v) if is_deg else v
            if func == 'sin':
                out = math.sin(arg)
            elif func == 'cos':
                out = math.cos(arg)
            else:
                out = math.tan(arg)
            print(f"{func}({v}{'°' if is_deg else ' rad'}) = {out}")
        else:
            # inverse trig outputs radians by default, convert to deg if user asked deg
            if func == 'asin':
                out_rad = math.asin(v)
            elif func == 'acos':
                out_rad = math.acos(v)
            else:
                out_rad = math.atan(v)
            out = math.degrees(out_rad) if is_deg else out_rad
            print(f"{func}({v}) = {out} {'°' if is_deg else 'rad'}")
    except Exception as e:
        print("Error in trig calculation:", e)

# --- number system conversion helpers ---

DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def int_to_base(n: int, base: int) -> str:
    """
    Convert integer n to string representation in given base (2..36).
    Works for negative numbers too.
    """
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")
    if n == 0:
        return "0"
    sign = '-' if n < 0 else ''
    n = abs(n)
    digits = []
    while n:
        digits.append(DIGITS[n % base])
        n //= base
    return sign + ''.join(reversed(digits))

def base_to_int(s: str, base: int) -> int:
    """
    Convert string s in given base to an integer.
    Accepts optional leading '-' for negative numbers.
    """
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36.")
    s = s.strip().upper()
    sign = -1 if s.startswith('-') else 1
    if s.startswith(('+', '-')):
        s = s[1:]
    value = 0
    for ch in s:
        if ch not in DIGITS[:base]:
            raise ValueError(f"Digit '{ch}' not valid for base {base}")
        value = value * base + DIGITS.index(ch)
    return sign * value

def number_system_conversion():
    """
    Convert integer numbers between bases (2..36).
    NOTE: this function handles integers only. Fractional conversions are not implemented.
    """
    print("\nNumber System Conversion (integers only).")
    s = input("Enter the number to convert (e.g. 1011 or FF or 123): ").strip()
    if s == '':
        print("No number entered. Returning.")
        return
    try:
        src_base_raw = input("Source base (2..36) (default 10): ").strip()
        src_base = int(src_base_raw) if src_base_raw else 10
        tgt_base_raw = input("Target base (2..36) (default 2): ").strip()
        tgt_base = int(tgt_base_raw) if tgt_base_raw else 2
        # convert input to integer decimal
        decimal_value = base_to_int(s, src_base)
        # convert decimal to target base string
        out = int_to_base(decimal_value, tgt_base)
        print(f"{s} (base {src_base}) = {out} (base {tgt_base})")
    except Exception as e:
        print("Conversion error:", e)


# --- plotting functions ---

def plot_2d():
    """
    Plot y = f(x) in 2D. User supplies function in x and range.
    """
    print("\n2D Plotting: y = f(x)")
    func_str = input("Enter function f(x), e.g. sin(x) + x**2: ").strip()
    if not func_str:
        print("No function entered. Returning.")
        return
    func_str = sanitize_input_expr(func_str)
    # get range
    x_min = float_input("x min (default -10): ", default=-10.0)
    x_max = float_input("x max (default 10): ", default=10.0)
    if x_max <= x_min:
        print("Invalid range: x_max must be greater than x_min.")
        return
    samples = int(float_input("Number of sample points (default 400): ", default=400))
    try:
        # Parse the symbolic expression
        x = sp.Symbol('x')
        f_expr = parse_sympy_expr(func_str)
        # Turn into a fast numeric function using numpy
        f_num = sp.lambdify(x, f_expr, modules=['numpy'])
        # Prepare x values
        xs = np.linspace(x_min, x_max, samples)
        # Evaluate y values; wrap in try because user functions may blow up
        ys = f_num(xs)
        # If ys is complex (some functions), try taking real part or fail
        if np.iscomplexobj(ys):
            print("Warning: function produced complex values. Plotting real parts.")
            ys = np.real(ys)
        # Plot using matplotlib
        plt.figure()
        plt.plot(xs, ys)
        plt.grid(True)
        plt.xlabel('x')
        plt.ylabel('y = f(x)')
        plt.title(f'y = {sp.pretty(f_expr)}')
        print("Displaying plot window... (close the window to continue)")
        plt.show()
    except Exception as e:
        print("Error while plotting 2D:", e)

def plot_3d():
    """
    Plot z = f(x, y) as a surface over given x & y ranges.
    """
    print("\n3D Plotting: z = f(x, y)")
    func_str = input("Enter function f(x,y), e.g. sin(x)*cos(y): ").strip()
    if not func_str:
        print("No function entered. Returning.")
        return
    func_str = sanitize_input_expr(func_str)
    x_min = float_input("x min (default -5): ", default=-5.0)
    x_max = float_input("x max (default 5): ", default=5.0)
    y_min = float_input("y min (default -5): ", default=-5.0)
    y_max = float_input("y max (default 5): ", default=5.0)
    if x_max <= x_min or y_max <= y_min:
        print("Invalid ranges. x_max > x_min and y_max > y_min required.")
        return
    samples = int(float_input("Grid samples per axis (default 50): ", default=50))
    try:
        # Create symbols and parse expression
        x, y = sp.symbols('x y')
        f_expr = parse_sympy_expr(func_str)
        # Create numeric function using numpy
        f_num = sp.lambdify((x, y), f_expr, modules=['numpy'])
        # Create meshgrid
        xs = np.linspace(x_min, x_max, samples)
        ys = np.linspace(y_min, y_max, samples)
        X, Y = np.meshgrid(xs, ys)
        Z = f_num(X, Y)
        if np.iscomplexobj(Z):
            print("Warning: function produced complex values. Plotting real parts.")
            Z = np.real(Z)
        # Plot surface
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Plotting the surface (rstride, cstride default fine)
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z = f(x,y)')
        ax.set_title(f"z = {sp.pretty(f_expr)}")
        print("Displaying 3D plot window... (close the window to continue)")
        plt.show()
    except Exception as e:
        print("Error while plotting 3D:", e)

# --- main menu loop ---

def print_menu():
    """
    Print the main menu options for the user.
    """
    print("\n===== Terminal Calculator Menu =====")
    print("1) Simple calculation (expressions)")
    print("2) Derivative calculation (symbolic)")
    print("3) Integration calculation (symbolic)")
    print("4) Trigonometry calculations")
    print("5) Number system conversion (integers)")
    print("6) 2D graph plotting (y = f(x))")
    print("7) 3D graph plotting (z = f(x,y))")
    print("0) Exit")
    print("====================================")

def main():
    """
    Main interactive loop. Calls feature functions based on user choice.
    The loop continues until user chooses to exit.
    """
    print("Welcome to the Terminal Calculator!")
    print("This tool uses sympy for symbolic math and matplotlib for plotting.")
    # Enter loop
    while True:
        try:
            print_menu()
            choice = input("Choose an option (0-7): ").strip()
            if choice == '1':
                simple_calculation()
            elif choice == '2':
                derivative_calculation()
            elif choice == '3':
                integration_calculation()
            elif choice == '4':
                trigonometry_calculation()
            elif choice == '5':
                number_system_conversion()
            elif choice == '6':
                plot_2d()
            elif choice == '7':
                plot_3d()
            elif choice == '0':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter a number between 0 and 7.")
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nKeyboard interrupt detected. Exiting.")
            break
        except Exception as e:
            # Catch-all to avoid program crash; show message and continue
            print("An unexpected error occurred:", e)

if __name__ == "__main__":
    main()



