"""
Patch pytrendline for pandas 2.x compatibility.
pandas 2.0 removed DataFrame.append() — replace with pd.concat().
"""
import pytrendline
import os

detect_path = os.path.join(os.path.dirname(pytrendline.__file__), "detect.py")

with open(detect_path, "r") as f:
    code = f.read()

# Replace df.append({...}, ignore_index=True) with pd.concat([df, pd.DataFrame([{...}])], ignore_index=True)
old = "trends_df = trends_df.append({"
new = "trends_df = pd.concat([trends_df, pd.DataFrame([{"

if old in code:
    code = code.replace(old, new)
    # Fix the closing: }, ignore_index=True) -> }])], ignore_index=True)
    code = code.replace("}, ignore_index=True)", "}])], ignore_index=True)")

    # Ensure pandas is imported
    if "import pandas as pd" not in code:
        code = "import pandas as pd\n" + code

    with open(detect_path, "w") as f:
        f.write(code)
    print(f"Patched {detect_path}")
else:
    print("Already patched or different version")
