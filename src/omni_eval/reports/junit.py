"""JUnit report for a completed evaluation, not metric recomputation."""

from xml.etree.ElementTree import Element, SubElement, tostring

from omni_eval.types.run import Run


class Junit:
    def render(self, run: Run) -> str:
        root = Element("testsuite", name="omni-eval", tests=str(len(run.scores)))
        properties = SubElement(root, "properties")
        SubElement(properties, "property", name="fixture", value=str(run.fixture).lower())
        if run.fixture:
            root.set("name", "omni-eval DEMO / SYNTHETIC")
        for score in run.scores:
            case = SubElement(root, "testcase", name=score.name)
            if score.status in {"error", "judge_error", "timeout", "blocked"}:
                SubElement(case, "error", message=score.status)
            elif score.status != "complete":
                SubElement(case, "skipped", message=score.status)
        return tostring(root, encoding="unicode")
