from setuptools import setup, find_packages

setup(
    name='trans4test',  # 包的名称
    version='0.1.0',  # 包的版本
    author='Your Name',  # 作者
    author_email='chenbojin@zju.edu.cn',  # 作者邮箱
    description='A machine learning model for XYZ',  # 简要描述
    long_description=open('README.md').read(),  # 详细描述（通常从 README.md 加载）
    long_description_content_type='text/markdown',  # 描述类型
    url='https://github.com/your_username/my_model',  # 项目地址
    packages=find_packages(),  # 自动发现所有子包
    install_requires=open('requirements.txt').read().splitlines(),  # 依赖
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 支持的 Python 版本
)
