import baostock as bs
import pandas as pd
import cfg

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
for i, head in enumerate(cfg.code):
    for j in range(3044):
        # 原有字符往右边调整，左边不足的位置补零
        print(head + str(j).rjust(4, "0"))
        code = head + str(j).rjust(4, "0")
        rs = bs.query_history_k_data_plus(code,
                                          "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,"
                                          "tradestatus, "
                                          "pctChg,isST",
                                          start_date='2022-3-13', end_date='2022-5-18',
                                          frequency="d", adjustflag="2")
        # print('query_history_k_data_plus respond error_code:' + rs.error_code)
        # print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)

        #### 结果集输出到csv文件 ####
        if len(result):
            # print(1)
            result.to_csv(r"..\..\source\stock\test\{}.csv".format(code), index=False, mode='w')
            # print(result)

#### 登出系统 ####
bs.logout()
