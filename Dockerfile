FROM public.ecr.aws/lambda/python:3.10

COPY exports ./exports
COPY hourlyExports ./hourlyExports
COPY hourlyExports/scalers ./hourlyExports/scalers
COPY hourlyExports/datas ./hourlyExports/scalers

# Copy requirements.txt
COPY equity_bse.csv ${LAMBDA_TASK_ROOT}
COPY equity_nse.csv ${LAMBDA_TASK_ROOT}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Copy function code
COPY main.py ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "main.handler" ]