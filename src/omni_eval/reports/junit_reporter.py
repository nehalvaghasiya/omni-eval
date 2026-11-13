from xml.etree.ElementTree import Element, SubElement, tostring

from omni_eval.schemas.run_result import RunResult


class JunitReporter:
    def render(self, run: RunResult) -> str:
        root = Element("testsuite", name="omni-eval", tests=str(len(run.metrics)))

        for metric in run.metrics:
            case = SubElement(root, "testcase", name=metric.name)

            if metric.status != "completed":
                SubElement(
                    case,
                    "skipped",
                    message=f"{metric.status}; coverage={metric.coverage}",
                )

        return tostring(root, encoding="unicode")
