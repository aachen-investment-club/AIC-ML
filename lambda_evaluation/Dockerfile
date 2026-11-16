# build from the AIC-ML root with:
# docker build --platform linux/amd64 --provenance=false -f lambda_evaluation/Dockerfile --build-context portfolio=../portfolio-management -t weekly-evaluation .

FROM public.ecr.aws/lambda/python:3.12

COPY lambda_evaluation/requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=portfolio portfolio/ ${LAMBDA_TASK_ROOT}/portfolio/
COPY strategies/ ${LAMBDA_TASK_ROOT}/strategies/
COPY lambda_evaluation/handler.py ${LAMBDA_TASK_ROOT}/lambda_evaluation/

CMD [ "lambda_evaluation.handler.handler" ]
