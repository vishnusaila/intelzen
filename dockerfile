FROM public.ecr.aws/lambda/python:3.12

# Copy requirements file
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# COPY function code
COPY Retizenapp.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler 
CMD ["Retizenapp.handler"]