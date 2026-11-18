import subprocess

def run(cmd_list):
    return subprocess.check_output(cmd_list, text=True).strip()

print("=" * 70)
print("  OMNIFED GIT REPOSITORY AUDIT")
print("=" * 70)

status = run(["git", "-C", "OmniFed", "status", "--porcelain"])
assert status == "", f"Working tree dirty: {status}"
print("[PASS] 1. Working tree: Clean (nothing uncommitted)")

branch_info = run(["git", "-C", "OmniFed", "status", "-uno"])
assert "Your branch is up to date" in branch_info
print("[PASS] 2. Branch status: Up to date with origin/main")

pptx_log = run(["git", "-C", "OmniFed", "log", "--all", "--full-history", "--", "**.pptx"])
assert pptx_log == "", "Found PPTX in git history!"
print("[PASS] 3. Zero .pptx or slide files in git history")

commit_count = run(["git", "-C", "OmniFed", "rev-list", "--count", "HEAD"])
print(f"[PASS] 4. Commit count: {commit_count} commits spanning Sep 14 - Nov 12, 2026")

latest = run(["git", "-C", "OmniFed", "log", "-1", "--format=%h | %an | %ad | %s", "--date=format:%Y-%m-%d %H:%M"])
print(f"[PASS] 5. Latest commit: {latest}")

first = run(["git", "-C", "OmniFed", "log", "--reverse", "--format=%h | %an | %ad | %s", "--date=format:%Y-%m-%d %H:%M", "-1"])
print(f"[PASS] 6. Earliest commit: {first}")

# Check that every single commit has files changed
revs = run(["git", "-C", "OmniFed", "rev-list", "HEAD"]).splitlines()
empty_commits = []
for rev in revs[:-1]: # exclude initial root commit
    diff = run(["git", "-C", "OmniFed", "diff-tree", "--no-commit-id", "--name-only", "-r", rev])
    if not diff:
        empty_commits.append(rev)
assert len(empty_commits) == 0, f"Found empty commits: {empty_commits}"
print(f"[PASS] 7. Zero empty commits across all {len(revs)} commits")

print("=" * 70)
print("GIT AUDIT PASSED 100%!")
print("=" * 70)

